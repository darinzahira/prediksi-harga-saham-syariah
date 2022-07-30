from .models import Predict
from . import db
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import timedelta, datetime
import math
import json
from matplotlib.pylab import *
from flask import Flask, Blueprint, render_template, request, flash, jsonify
from flask_login import login_required, current_user
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
matplotlib.use('Agg')

views = Blueprint('views', __name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'helloworld'


@views.route('/')
def home():
    return render_template("input.html", user=current_user)


@views.route('/khusus')
def homekhusus():
    return render_template("form.html", user=current_user)


@views.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':

        # Mengambil data dari yahoo finance
        symbol = request.form.get("symbol")
        target = request.form.get("target")
        end = datetime.datetime.today() + timedelta(1)
        steps = 0

        # Mendapatkan start date untuk prediksi
        def startdate(from_date, add_days):
            business_days_to_add = add_days + 20
            current_date = from_date
            while business_days_to_add > 0:
                current_date -= datetime.timedelta(days=1)
                weekday = current_date.weekday()
                if weekday >= 5:  # sunday = 6
                    continue
                business_days_to_add -= 1
            return current_date

        if len(symbol) < 1:
            flash('Simbol emiten harus dipilih!', category='error')
        elif len(target) < 1:
            flash('Tanggal target harus diisi!', category='error')
        else:
            if (symbol == 'ANTM.JK'):
                steps += 25
                model = {'lstm': 'ANTM2_epoch100_ts25'}
                emiten = "Aneka Tambang Tbk"
            elif (symbol == 'ERAA.JK'):
                steps += 100
                model = {'lstm': 'ERAA2_epoch50_ts100'}
                emiten = "Erajaya Swasembada Tbk"
            elif (symbol == 'KLBF.JK'):
                steps += 25
                model = {'lstm': 'KLBF2_epoch100_ts25'}
                emiten = "Kalbe Farma Tbk"
            elif (symbol == 'WIKA.JK'):
                steps += 75
                model = {'lstm': 'WIKA_epoch100_ts75'}
                emiten = "Wijaya Karya (Persero) Tbk"
            elif (symbol == 'SMGR.JK'):
                steps += 75
                model = {'lstm': 'SMGR_epoch100_ts75'}
                emiten = "Semen Indonesia Tbk"
            elif (symbol == 'ADRO.JK'):
                steps += 100
                model = {'lstm': 'ADRO_epoch100_ts100'}
                emiten = "Adaro Energy Indonesia Tbk"
            elif (symbol == 'INDF.JK'):
                steps += 50
                model = {'lstm': 'INDF_epoch100_ts50'}
                emiten = "Indofood Sukses Makmur Tbk"
            elif (symbol == 'MNCN.JK'):
                steps += 25
                model = {'lstm': 'MNCN_epoch100_ts25'}
                emiten = "Media Nusantara Citra Tbk"
            elif (symbol == 'PGAS.JK'):
                steps += 75
                model = {'lstm': 'PGAS_epoch100_ts75'}
                emiten = "Perusahaan Gas Negara Tbk"
            elif (symbol == 'TLKM.JK'):
                steps += 25
                model = {'lstm': 'TLKM_epoch100_ts25'}
                emiten = "Telkom Indonesia (Persero) Tbk"

            data = yf.download(symbol, start=startdate(end, steps),
                               end=end, progress=False)
            lstm = load_model(
                'website\\static\\model\\{lstm}.h5'.format_map(model))

            # Preprocessing Data dengan MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            data = data.filter(['Close'])
            dataset = data[-steps:].values
            scaled_data = scaler.fit_transform(dataset)

            # Menghitung jumlah hari kerja
            date_target = datetime.datetime.strptime(target, '%Y-%m-%d')

            def daterange(date1, date2):
                for n in range(int((date2 - date1).days)+2):
                    yield date1 + timedelta(n)

            start_dt = end
            end_dt = date_target

            dates = []
            day = 0

            weekdays = [6, 7]
            for dt in daterange(start_dt, end_dt):
                if dt.isoweekday() not in weekdays:
                    dates.append(dt.strftime("%Y-%m-%d"))
                    day = day+1
            dates = np.array(dates)

            # Reshape dan membuat list
            xinput = scaled_data.reshape(1, -1)
            temp_input = list(xinput)
            temp_input = temp_input[0].tolist()

            # Prediksi harga saham
            lst_output = []
            n_steps = steps
            i = 0
            while(i < day):

                if(len(temp_input) > steps):
                    x_input = np.array(temp_input[1:])
                    x_input = x_input.reshape(1, -1)
                    x_input = x_input.reshape((1, n_steps, 1))
                    yhat = lstm.predict(x_input, verbose=0)
                    temp_input.extend(yhat[0].tolist())
                    temp_input = temp_input[1:]
                    lst_output.extend(yhat.tolist())
                    i = i+1
                else:
                    x_input = xinput.reshape((1, n_steps, 1))
                    yhat = lstm.predict(x_input, verbose=0)
                    temp_input.extend(yhat[0].tolist())
                    lst_output.extend(yhat.tolist())
                    i = i+1

            # Denormalisasi
            lst_output = scaler.inverse_transform(lst_output)
            price = lst_output[day-1]
            targetdate = dates[day-1]

            predict = []
            for i in range(0, len(dates)):
                predict.append(
                    {"Date": dates[i], "Predictions": lst_output[i]})

            # Grafik
            plt.figure(figsize=(8, 4))
            plt.plot(lst_output, 'r', label='Harga Prediksi')
            plt.ylabel('Prediksi')
            plt.xlabel('Index')
            plt.legend()
            plt.savefig('website\\static\\images\\result_'+symbol+'.png')

            return render_template("result_predict.html", context=predict, symbol=symbol, target=targetdate, price=price, user=current_user, emiten=emiten)

    return render_template("form.html", user=current_user)


@views.route('/formm', methods=['GET', 'POST'])
def formm():
    if request.method == 'POST':

        # Mengambil data dari yahoo finance
        symbol = request.form.get("symbol")
        start = request.form.get("start")
        end = request.form.get("end")
        steps = 0

        if len(symbol) < 1:
            flash('Simbol emiten harus dipilih!', category='error')
        elif len(start) < 1:
            flash('Tanggal awal harus diisi!', category='error')
        elif len(end) < 1:
            flash('Tanggal akhir harus diisi!', category='error')
        else:
            if (symbol == 'ANTM.JK'):
                steps += 25
                model = {'lstm': 'ANTM2_epoch100_ts25'}
                emiten = "Aneka Tambang Tbk"
            elif (symbol == 'ERAA.JK'):
                steps += 100
                model = {'lstm': 'ERAA2_epoch50_ts100'}
                emiten = "Erajaya Swasembada Tbk"
            elif (symbol == 'KLBF.JK'):
                steps += 25
                model = {'lstm': 'KLBF2_epoch100_ts25'}
                emiten = "Kalbe Farma Tbk"
            elif (symbol == 'WIKA.JK'):
                steps += 75
                model = {'lstm': 'WIKA_epoch100_ts75'}
                emiten = "Wijaya Karya (Persero) Tbk"
            elif (symbol == 'SMGR.JK'):
                steps += 75
                model = {'lstm': 'SMGR_epoch100_ts75'}
                emiten = "Semen Indonesia Tbk"
            elif (symbol == 'ADRO.JK'):
                steps += 100
                model = {'lstm': 'ADRO_epoch100_ts100'}
                emiten = "Adaro Energy Indonesia Tbk"
            elif (symbol == 'INDF.JK'):
                steps += 50
                model = {'lstm': 'INDF_epoch100_ts50'}
                emiten = "Indofood Sukses Makmur Tbk"
            elif (symbol == 'MNCN.JK'):
                steps += 25
                model = {'lstm': 'MNCN_epoch100_ts25'}
                emiten = "Media Nusantara Citra Tbk"
            elif (symbol == 'PGAS.JK'):
                steps += 75
                model = {'lstm': 'PGAS_epoch100_ts75'}
                emiten = "Perusahaan Gas Negara Tbk"
            elif (symbol == 'TLKM.JK'):
                steps += 25
                model = {'lstm': 'TLKM_epoch100_ts25'}
                emiten = "Telkom Indonesia (Persero) Tbk"

            data = yf.download(symbol, start=start,
                               end=end, progress=False)
            newdata = data.reset_index()
            newdata['Date'] = newdata['Date'].dt.strftime('%Y-%m-%d')

            # Plot harga saham
            plt.figure(figsize=(8, 4))
            plt.plot(data.Close)
            plt.savefig('website\\static\\images\\price_'+symbol+'.png')

            # Ubah data dari dataframe menjadi json
            json_records = newdata.to_json(orient='records')
            data_json = []
            data_json = json.loads(json_records)

            # Preprocessing Data dengan MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            data = newdata.filter(['Close'])
            dataset = data.values
            scaled_data = scaler.fit_transform(dataset)

            # Pembagian Data latih dan Data uji
            training_data_len = math.ceil((len(dataset) * .8))

            # Membagi data training sesuai dengan timesteps
            train_data = scaled_data[0:training_data_len, :]
            x_train = []
            y_train = []

            for i in range(steps, len(train_data)):
                x_train.append(train_data[i-steps:i, 0])
                y_train.append(train_data[i, 0])

            # Ubah x_train dan y_train ke numpy array
            x_train, y_train = np.array(x_train), np.array(y_train)
            x_train = np.reshape(
                x_train, (x_train.shape[0], x_train.shape[1], 1))

            # Load model yang telah disimpan
            lstm = load_model(
                'website\\static\\model\\{lstm}.h5'.format_map(model))

            # Prediksi harga saham dengan data training
            predict = lstm.predict(x_train)
            predict = scaler.inverse_transform(predict)

            train = data[steps:training_data_len]
            train['Predictions'] = predict
            train['Date'] = newdata['Date']

            train_record = train.to_json(orient='records')
            train_json = []
            train_json = json.loads(train_record)

            predictPlot = np.empty_like(scaled_data)
            predictPlot[:, :] = np.nan
            predictPlot[steps:len(predict)+steps, :] = predict

            # MSE dan MAPE train
            mse_train = mean_squared_error(y_train, predict)
            mape_train = mean_absolute_percentage_error(y_train, predict)

            # plot hasil prediksi dengan train data
            plt.figure(figsize=(8, 4))
            plt.plot(scaler.inverse_transform(scaled_data))
            plt.plot(predictPlot)
            plt.savefig('website\\static\\images\\predict_'+symbol+'.png')

            # Membagi data testing sesuai dengan timesteps
            test_data = scaled_data[training_data_len - steps:, :]
            x_test = []
            y_test = dataset[training_data_len:, :]
            for i in range(steps, len(test_data)):
                x_test.append(test_data[i-steps:i, 0])

            # Ubah x_test ke numpy array
            x_test = np.array(x_test)
            x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

            # Prediksi dengan menggunakan data testing
            predictions = lstm.predict(x_test)
            predictions = scaler.inverse_transform(predictions)
            train = data[:training_data_len]
            valid = data[training_data_len:]
            valid['Predictions'] = predictions
            valid['Date'] = newdata['Date']

            json_record = valid.to_json(orient='records')
            valid_json = []
            valid_json = json.loads(json_record)

            # MSE dan MAPE test
            mse_test = mean_squared_error(y_test, predictions)
            mape_test = mean_absolute_percentage_error(y_test, predictions)

            # Plot hasil prediksi dengan test data
            plt.figure(figsize=(8, 4))
            plt.plot(train['Close'])
            plt.plot(valid[['Close', 'Predictions']])
            plt.legend(['Train', 'Actual', 'Predictions'], loc='lower right')
            plt.savefig('website\\static\\images\\predicted_'+symbol+'.png')

            # Plot hasil test data
            plt.figure(figsize=(8, 4))
            plt.plot(y_test, 'b', label='Actual Price')
            plt.plot(predictions, 'r', label='Predicted Price')
            plt.ylabel('Price')
            plt.legend()
            plt.savefig('website\\static\\images\\result_'+symbol+'.png')

            return render_template("predict.html", context=data_json, symbol=symbol, emiten=emiten,
                                   valid=valid_json, train=train_json, user=current_user, mse_train=mse_train,
                                   mse_test=mse_test, mape_test=mape_test, mape_train=mape_train)

    return render_template("form.html", user=current_user)


@views.route('/save/<target>/<price>/<emiten>', methods=['GET', 'POST'])
@login_required
def save(target, price, emiten):
    target = target
    price = price
    emiten = emiten

    new_predict = Predict(emiten=emiten, target=target, predict=price,
                          user_id=current_user.id)
    db.session.add(new_predict)
    db.session.commit()
    flash('Data berhasil disimpan!', category='success')
    return render_template("histori.html", user=current_user)


@views.route('/histori')
@login_required
def histori():
    return render_template("histori.html", user=current_user)


@views.route('/delete-predict', methods=['POST'])
@login_required
def delete_predict():
    predict = json.loads(request.data)
    predictId = predict['predictId']
    predict = Predict.query.get(predictId)
    if predict:
        if predict.user_id == current_user.id:
            db.session.delete(predict)
            db.session.commit()
            flash('Data berhasil dihapus', category='error')
    return jsonify({})
