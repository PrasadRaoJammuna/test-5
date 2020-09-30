from flask import Flask,render_template,request
import pickle
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)


colleges = [1026, 1545, 1550, 1038, 18, 1555, 1557, 1046, 1562, 32, 1571, 1575, 1580, 45, 54, 57, 1593, 59, 61, 63, 1600, 67, 72, 73, 80, 1629, 95, 103, 1645, 110, 1647, 111, 113, 628, 117, 116, 119, 634, 123, 1149, 130, 1666, 132, 1669, 1667, 144, 146, 155, 157, 158, 1182, 166, 1703, 183, 193, 195, 197, 199, 204, 205, 213, 731, 222, 742, 231, 232, 1772, 1774, 241, 1778, 244, 249, 764, 771, 1803, 787, 1813, 815, 313, 327, 379, 895, 403, 922, 416, 1441, 1451, 433, 436, 1460, 444, 453, 1477, 1478, 975, 469, 982, 474, 475, 476, 989, 1509, 1006]


@app.route('/')
def hello_world():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    
    if request.method =='POST':
        
        #pickle_in = open("corona.pkl","rb")
        
        # GET Data from FORM
        #try:
            year = float(request.form['year'])
            gender = float(request.form['gender'])
            ethnicity = float(request.form['ethnicity'])
            major = float(request.form['major'])
            school = float(request.form['school'])
            location = float(request.form['location'])
            ib = float(request.form['ib'])
            gpaun = float(request.form['gpaun'])
            gpaw = float(request.form['gpaw'])
            satmath = float(request.form['satmath'])
            satwriting = float(request.form['satwriting'])
            satrw = float(request.form['satrw'])
            satread = float(request.form['satread'])
            satwrite = float(request.form['satwrite'])
            scorefrom = float(request.form['scorefrom'])
            satact = float(request.form['satact'])
            heng = float(request.form['heng'])
            hmath = float(request.form['hmath'])
            hscience = float(request.form['hscience'])
            hoteng = float(request.form['hoteng'])
            hsocial = float(request.form['hsocial'])
            hvart = float(request.form['hvart'])
            hact = float(request.form['hact'])
            

            #print('#-------------------------------data is here-------------------------------------#')
            clf = pickle.load(open("rf_model.pkl", "rb"))
            
            pb_scores = []
            status  = []
            for c_id in colleges:
                data = [[year,gender,ethnicity,major,school,location,ib,gpaun,gpaw,satmath,satwriting,satrw,satread,satwrite,scorefrom,satact,heng,hmath,hscience,hoteng,hsocial,hvart,hact,c_id]]

                #print(len(data),10*'-----')
                prediction = clf.predict(data)[0]
                proba_score = clf.predict_proba(data)[0][0]
                print('college:',c_id,'Status:',prediction,10*'-----')
                status.append(prediction)
                pb_scores.append(proba_score)
                print()

            context = zip(colleges,status,pb_scores)
            
            return render_template('predict.html',results = context)
        #except Exception as e:
            #print('Something wrong------------------------')
            #return render_template('predict.html',status = 'Something went wrong.!')
            
    else:
        
        return render_template('index.html',message='Something missed, Please follow the instructions..!')
              

if __name__ == '__main__':
    app.run(debug=False)