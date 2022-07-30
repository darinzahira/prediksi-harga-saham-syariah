from flask import Blueprint, render_template, request, flash, redirect, url_for
from .models import User
from werkzeug.security import generate_password_hash, check_password_hash
from . import db
from flask_login import login_user, login_required, logout_user, current_user

auth = Blueprint('auth', __name__)


@auth.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        first_name = request.form.get('first_name')
        password = request.form.get('password')

        if first_name == "userkhusus":
            user = User.query.filter_by(first_name=first_name).first()
            if user:
                if check_password_hash(user.password, password):
                    flash('Berhasil Login!', category='success')
                    login_user(user, remember=True)
                    return redirect(url_for('views.homekhusus'))
                else:
                    flash('Password salah, silahkan coba lagi.', category='error')
            else:
                flash('Username tidak terdaftar, silahkan coba lagi.',
                      category='error')
        else:
            user = User.query.filter_by(first_name=first_name).first()
            if user:
                if check_password_hash(user.password, password):
                    flash('Berhasil Login!', category='success')
                    login_user(user, remember=True)
                    return redirect(url_for('views.home'))
                else:
                    flash('Password salah, silahkan coba lagi.', category='error')
            else:
                flash('Username tidak terdaftar, silahkan coba lagi.',
                      category='error')

    return render_template("login.html", user=current_user)


@auth.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))


@auth.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email')
        first_name = request.form.get('firstName')
        password1 = request.form.get('password1')
        password2 = request.form.get('password2')

        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email telah terdaftar.', category='error')
        elif len(email) < 4:
            flash('Email harus lebih dari 3 karakter.', category='error')
        elif len(first_name) < 2:
            flash('Username harus lebih dari 1 character.', category='error')
        elif password1 != password2:
            flash('Password tidak cocok.', category='error')
        elif len(password1) < 7:
            flash('Password minimal 7 karakter.', category='error')
        else:
            new_user = User(email=email, first_name=first_name, password=generate_password_hash(
                password1, method='sha256'))
            db.session.add(new_user)
            db.session.commit()
            login_user(new_user, remember=True)
            flash('Akun berhasil dibuat!', category='success')
            return redirect(url_for('views.home'))

    return render_template("sign_up.html", user=current_user)
