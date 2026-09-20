from __future__ import annotations

import typer

from ragfabric_cli.commands.common import collection_by_name, session, user_by_email
from ragfabric_core.auth import service
from ragfabric_core.auth.api_keys import create_api_key
from ragfabric_core.models.access import ApiKey, CollectionGrant, Group
from ragfabric_core.models.document import Collection

groups_app = typer.Typer(help="Groups and membership.")
grants_app = typer.Typer(help="Collection grants.")
keys_app = typer.Typer(help="API keys.")


def _group(db, name: str) -> Group:
    group = db.query(Group).filter(Group.name == name).first()
    if group is None:
        typer.echo(f"group not found: {name}")
        raise typer.Exit(code=1)
    return group


@groups_app.command("create")
def create_group(name: str, description: str = typer.Option("")) -> None:
    with session() as db:
        if db.query(Group).filter(Group.name == name).first() is not None:
            typer.echo(f"group already exists: {name}")
            raise typer.Exit(code=1)
        service.create_group(db, name, description)
        db.commit()
    typer.echo(f"created group {name}")


@groups_app.command("add-member")
def add_member(group_name: str, email: str) -> None:
    with session() as db:
        group = _group(db, group_name)
        user = user_by_email(db, email)
        service.add_member(db, group.id, user.id)
        db.commit()
    typer.echo(f"added {email} to {group_name}")


@groups_app.command("list")
def list_groups() -> None:
    with session() as db:
        for g in db.query(Group).order_by(Group.name).all():
            members = service_members(db, g.id)
            typer.echo(f"{g.id}\t{g.name}\t{members} member(s)")


def service_members(db, group_id: int) -> int:
    from ragfabric_core.models.access import GroupMember

    return db.query(GroupMember).filter(GroupMember.group_id == group_id).count()


@grants_app.command("add")
def add_grant(
    group: str = typer.Option(...),
    collection: str = typer.Option(...),
    permission: str = typer.Option("read"),
) -> None:
    with session() as db:
        g = _group(db, group)
        c = collection_by_name(db, collection)
        try:
            service.grant_collection(db, g.id, c.id, permission)
        except ValueError as exc:
            typer.echo(str(exc))
            raise typer.Exit(code=1) from None
        db.commit()
    typer.echo(f"granted {permission} on {collection} to {group}")


@grants_app.command("list")
def list_grants() -> None:
    with session() as db:
        rows = (
            db.query(CollectionGrant, Group.name, Collection.name)
            .join(Group, Group.id == CollectionGrant.group_id)
            .join(Collection, Collection.id == CollectionGrant.collection_id)
            .all()
        )
        for grant, group_name, collection_name in rows:
            typer.echo(f"{grant.id}\t{group_name}\t{collection_name}\t{grant.permission}")


@keys_app.command("create")
def create_key(
    name: str = typer.Option(...),
    user: str = typer.Option(...),
    collection: list[str] = typer.Option([], "--collection"),
    strategy: list[str] = typer.Option([], "--strategy"),
    rate_limit: int | None = typer.Option(
        None,
        "--rate-limit",
        help="Requests/minute; defaults to limits.rate_limit_per_minute in ragfabric.yaml.",
    ),
) -> None:
    with session() as db:
        owner = user_by_email(db, user)
        collection_ids = [collection_by_name(db, c).id for c in collection]
        key, plaintext = create_api_key(
            db,
            name=name,
            user_id=owner.id,
            collection_ids=collection_ids,
            strategies=strategy,
            rate_limit_per_minute=rate_limit,
        )
        db.commit()
        typer.echo(f"created key {key.id} ({name}) for {user}")
    typer.echo("store this now, it is not shown again:")
    typer.echo(plaintext)


@keys_app.command("list")
def list_keys() -> None:
    with session() as db:
        for k in db.query(ApiKey).order_by(ApiKey.id).all():
            typer.echo(
                f"{k.id}\t{k.name}\t{k.key_prefix}...\t{'active' if k.is_active else 'revoked'}\tlimit {k.rate_limit_per_minute}/min"
            )


@keys_app.command("revoke")
def revoke_key(key_id: int) -> None:
    with session() as db:
        key = db.get(ApiKey, key_id)
        if key is None:
            typer.echo(f"key not found: {key_id}")
            raise typer.Exit(code=1)
        key.is_active = False
        db.commit()
    typer.echo(f"revoked key {key_id}")
