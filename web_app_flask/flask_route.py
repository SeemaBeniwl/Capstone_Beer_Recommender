from flask import Flask, render_template, request, jsonify, url_for
from Content_based_engine.content_based_model import *
from CF.cf_final import *
import json
import pickle
import numpy as np

app = Flask(__name__)

@app.route("/")
@app.route("/index.html")
def index():
    return render_template("index.html")
@app.route("/content.html", methods = ["GET", "POST"])
def content():
    with open("./Content_based_engine/beer_list.p", 'rb') as bf:
        beer_list = pickle.load(bf)
    with open("./Conten_based_engine/index.p", "rb") as ix:
        index = pickle.load(ix)
    with open("./Content_based_engine/beer_keywords.json", "rb") as bk:
        beer_keywords = json.load(bk)
    with open("./Content_based_engine/dict_for_CB_table.p", "rb") as diCB:
        dict_for_cb_table = pickle.load(diCB)

    if request.method == "GET":
        return render_template("content.html" , beerlist = beer_list)
    beer_input = None
    if request.method == "POST" and "beer_input" in request.form:
        beer_input = request.form["beer_input"]
        if beer_input == '':
            return render_template("content.html", beerlist = beer_list)
        else:
            cb_rec = similar_beers(beer_input, beer_list, index, ntop=10)
            print cb_rec
            table_list = map(lambda x:dict_for_cb_table.get(x), cb_rec)
            key_words = beer_keywords[beer_input]
            key_words = map(lambda x: x.decode('utf-8', 'ignore').encode('ascii', 'ignore'), key_words)
            key_words = ', '.join(key_words)
            return render_template("content.html", beerlist=beer_list, cb_rec=cb_rec,
                                   table_list=table_list, key_words=key_words)


@app.route("/collab.html", methods=["GET", "POST"])
def collab():
    with open("./CF/beer_dict.pickle", "r") as bf:
        beer_dict = pickle.load(bf)
    beer_list = sorted(beer_dict.values())
    beer_list = [x.decode('ascii', 'ignore') for x in beer_list if '\x8a\x97\xc8' not in x] # we need to fix this later
    ratings_mat = np.load("./CF/ratings_mat.npy")
    ratings_mat, g_avg = cf_prerprocess(ratings_mat)

    if request.method == "POST":
        inp_tup = []
        for i in range(1, 6):
            beer_inp_key = "beer_inp" + str(i)
            rating_inp_key = "rating_inp" + str(i)
            if request.form[beer_inp_key] and request.form[rating_inp_key]:
                inp_tup.append((request.form[beer_inp_key], request.form[rating_inp_key]))

        if inp_tup == []:
            return render_template("collab.html", beer_list=beer_list)
        else:
            user_data = user_cf_proc(inp_tup, ratings_mat, beer_dict)
            # print user_data
            CF_rec = cf_rec(user_data, ratings_mat, g_avg, beer_dict)
            # print CS_rec

            return render_template("collab.html", cf_rec=CF_rec, beer_list=beer_list)

    else:
        return render_template("collab.html", beer_list=beer_list)

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    search = request.args.get('q')
    results = ['Beer1', 'Wine1', 'Soda1', 'a', 'b']
    return jsonify(matching_results=results)

if __name__ == "__main__":
    app.run(debug=True)


