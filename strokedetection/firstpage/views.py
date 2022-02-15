from os import write
from django.shortcuts import render
from django.http import HttpResponse
import pandas as pd


# Create your views here.

import joblib
modelReload=joblib.load('./models/RFModelforStroke.pkl')

def index(request):
    context={'a':'HelloWorld!'}
    return render(request, 'index.html',context)
    #return HttpResponse({'a':1})


def predictstroke(request):
    print (request)
    if request.method == 'POST':
        temp={}
        temp['gender']=request.POST.get('gender')
        temp['age']=request.POST.get('age')
        temp['heart_disease']=request.POST.get('heart_disease')
        temp['hypertension']=request.POST.get('hypertension')
        temp['ever_married']=request.POST.get('ever_married')
        temp['work_type']=request.POST.get('work_type')
        temp['residence_type']=request.POST.get('residence_type')
        temp['avg_glucose_level']=request.POST.get('avg_glucose_level')
        temp['bmi']=request.POST.get('bmi')
        temp['smoking_status']=request.POST.get('smoking_status')
    
    testDtaa = pd.DataFrame(temp, index=[0])
    scoreval = modelReload.predict(testDtaa)[0]
    
    
    context={'scoreval':scoreval,
            'temp':temp
    }
    return render(request, 'deneme.html',context)
    