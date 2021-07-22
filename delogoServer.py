from cv2 import data
from flask import Flask,request,render_template
import json
import requests
from delogo import Delogo
import ast
app = Flask(__name__)
delogo=Delogo()
'''
json={
    'method':'delogo',
    'path':'url',
    'maskcoords':'(0,0)',
    'maskRec':'(160,160)',
    'frameInterval':'(0,0)',
    'captionMode':'True',
    'patchSplit':'True',
    'extractFlag':'True',
}
'''
@app.route('/delogo/',methods=(['POST']))
def delogoApp():
    # print("i am here")
    rawData=request.get_data()
    data=json.loads(rawData)
    # data=request.form.to_dict()
    if 'path' in data.keys():
        path=data['path']
    else:
        return "bad"
    print(path)
    if 'maskcoords' in data.keys():
        maskcoords=ast.literal_eval(data['maskcoords'])
    else:
        maskcoords=(0,0)

    if 'maskRec' in data.keys():
        maskRec=ast.literal_eval(data['maskRec'])
    else:
        maskRec=(160,160)

    if 'frameInterval' in data.keys():
        frameInterval=ast.literal_eval(data['frameInterval'])
    else:
        frameInterval=(0,0)

    if 'cutCoords' in data.keys():
        cutCoords=ast.literal_eval(data['cutCoords'])
    else:
        cutCoords=None

    if 'cutRec' in data.keys():
        cutRec=ast.literal_eval(data['cutRec'])
    else:
        cutRec=None
    
    if data['method']=='delogo':
        if 'captionMode' in data.keys():
            captionMode=ast.literal_eval(data['captionMode'])
        else:
            captionMode=False
        if 'patchSplit' in data.keys():
            patchSplit=ast.literal_eval(data['patchSplit'])
        else:
            patchSplit=False
        if 'extractFlag' in data.keys():
            extractFlag=ast.literal_eval(data['extractFlag'])
        else:
            extractFlag=False
        delogo.run(path=path,maskcoords=maskcoords,maskRec=maskRec,frameInterval=frameInterval,output='./output.mp4',extractFlag=extractFlag,captionMode=captionMode,patchSplit=patchSplit,cutCoords=cutCoords,cutRec=cutRec)
    elif data['method']=='preview':
        delogo.run(path=path,maskcoords=maskcoords,maskRec=maskRec,frameInterval=frameInterval,output='./output.png',cutCoords=cutCoords,cutRec=cutRec)
    return "good"
@app.route("/test/")
def sendRequest():
    url = 'http://172.17.0.92:3487/delogo/'
    form={
    'method':'delogo',
    'path':'./lena2.mp4',
    'maskcoords':'(0,0)',
    'maskRec':'(160,160)',
    'frameInterval':'(0,10)',
    'captionMode':'True',
    'patchSplit':'True',
    'extractFlag':'True',
    # 'cutCoords':'(0,0)',
    # 'cutRec':'(320,320)'
    }
    s = json.dumps(form)
    r = requests.post(url, data=s)
    print(r.status_code)
    return "send post"
# @app.route("/")
# def home():
#     return render_template('delogo.html')
if __name__ == '__main__':   
    app.run(host="0.0.0.0",debug=False,port=3487)
    
    
