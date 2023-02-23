import threading
import webbrowser
from flask import Flask, render_template
import os


app = Flask(__name__)

def LoadViewer():
    return render_template('ViewAnimation.html')



def GetAllPossibleAnimations():
    AllSeeds = ''
    for filename in os.listdir(SaveSelectedMotionLocation):
        if(filename[len(filename)-1] == 'h'):
         AllSeeds += filename[0:len(filename)-4] + ','
    return AllSeeds

if __name__ == "__main__":
    SaveSelectedMotionLocation = './clustering/static/animations/'
    anims = GetAllPossibleAnimations()

    port = 23336
    host_name = "0.0.0.0"

    app.add_url_rule('/', 'index', LoadViewer)
    threading.Thread(
        target=lambda: app.run(host=host_name, port=port, debug=True, use_reloader=False)).start()
    webbrowser.open("http://127.0.0.1:23336/?AllSeedsDetected=" + str(GetAllPossibleAnimations()))

