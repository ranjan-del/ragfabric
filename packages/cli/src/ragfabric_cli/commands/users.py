from __future__ import annotations

import typer

from ragfabric_cli.commands.common import session, user_by_email
from ragfabric_core.models.user import Role, User
from ragfabric_core.security import hash_password

app = typer.Typer(help="Users.")


@app.command("create")
def create(
    email: str = typer.Option(...),
    password: str = typer.Option(..., prompt=False),
    role: str = typer.Option("user"),
) -> None:
    if role not in {Role.ADMIN.value, Role.USER.value}:
        typer.echo("role must be user or admin")
        raise typer.Exit(code=1)
    with session() as db:
        if db.query(User).filter(User.email == email.lower()).first() is not None:
            typer.echo(f"user already exists: {email}")
            raise typer.Exit(code=1)
        db.add(User(email=email.lower(), hashed_password=hash_password(password), role=role))
        db.commit()
    typer.echo(f"created {email} ({role})")


@app.command("list")
def list_users() -> None:
    with session() as db:
        for u in db.query(User).order_by(User.id).all():
            typer.echo(f"{u.id}\t{u.email}\t{u.role}\t{'active' if u.is_active else 'inactive'}")


@app.command("set-role")
def set_role(email: str, role: str) -> None:
    if role not in {Role.ADMIN.value, Role.USER.value}:
        typer.echo("role must be user or admin")
        raise typer.Exit(code=1)
    with session() as db:
        user = user_by_email(db, email)
        user.role = role
        db.commit()
    typer.echo(f"{email} is now {role}")


@app.command("deactivate")
def deactivate(email: str) -> None:
    with session() as db:
        user = user_by_email(db, email)
        user.is_active = False
        db.commit()
    typer.echo(f"{email} deactivated")
