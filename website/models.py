from . import db
from flask_login import UserMixin
from sqlalchemy.sql import func


class Predict(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    date = db.Column(db.DateTime(timezone=True), default=func.now())
    emiten = db.Column(db.String(100))
    predict = db.Column(db.String(100))
    target = db.Column(db.String(100))
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'))


class User(db.Model, UserMixin):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))
    first_name = db.Column(db.String(150))
    predicts = db.relationship('Predict')
