from __future__ import annotations

from contextlib import contextmanager

import typer
from sqlalchemy.orm import Session

from ragfabric_core.models.document import Collection
from ragfabric_core.models.user import User


@contextmanager
def session():
    from ragfabric_core.runtime import get_session_factory

    db: Session = get_session_factory()()
    try:
        yield db
    finally:
        db.close()


def user_by_email(db: Session, email: str) -> User:
    user = db.query(User).filter(User.email == email.lower()).first()
    if user is None:
        typer.echo(f"user not found: {email}")
        raise typer.Exit(code=1)
    return user


def collection_by_name(db: Session, name: str, create: bool = False) -> Collection:
    collection = db.query(Collection).filter(Collection.name == name).first()
    if collection is None:
        if not create:
            typer.echo(f"collection not found: {name}")
            raise typer.Exit(code=1)
        collection = Collection(name=name)
        db.add(collection)
        db.flush()
    return collection
