from flask import Flask, request
from ml import ml
import mysql.connector

app = Flask(__name__)

@app.route('/submit', methods=['POST'])
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
            result.append(data)
            cursor.execute(query, (data[0], data[1], data[3]))
            cnx.commit()

        cursor.close()
        cnx.close()
    except:
        result = 'error'
    finally:
        return {'data': result}
