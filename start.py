from flask import Flask, request
from ml import ml
import mysql.connector
from flask_cors import CORS, cross_origin


app = Flask(__name__)
cors = CORS(app)

@app.route('/submit', methods=['POST'])
@cross_origin()
def submit():
    result = []
    body = request.get_json()
    body_list = body['data']

    try:
        cnx = mysql.connector.connect(user='ukhfjo3tarnlmv5h', 
                                      password='e26FWW2pC9fZA34VHh2x',
                                      host='bg54pj3hyknizs3qytu9-mysql.services.clever-cloud.com',
                                      database='bg54pj3hyknizs3qytu9')
        
        cursor = cnx.cursor(prepared=True)
        query = """INSERT INTO feedback (detail, class, probability, datetime) VALUES (%s, %s, %s, NOW())"""

        for i in body_list:
            data = ml.main(i)
            cursor.execute(query, (data["INPUT"], data["CLASS_NO"], data["PROBABILITY"]))
            cnx.commit()
            data["PROBABILITY"] = f'{data["PROBABILITY"]:.2f}%'
            result.append(data)

        cursor.close()
        cnx.close()
    except Exception as e:
        result = str(e)
    finally:
        return {'data': result}
