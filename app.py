from flask import Flask, redirect,url_for,render_template, request
import nltk
import string
import model
dict_mapping=model.mapping

stopwords=['অতএব','অথচ','অথবা','অনুযায়ী','অনেক','অনেকে','অনেকেই','অন্তত','অন্য','অবধি','অবশ্য','অর্থাত','আই','আগামী','আগে','আগেই','আছে','আজ','আদ্যভাগে','আপনার','আপনি',
 'আবার',
 'আমরা',
 'আমাকে',
 'আমাদের',
 'আমার',
 'আমি',
 'আর',
 'আরও',
 'ই',
 'ইত্যাদি',
 'ইহা',
 'উচিত',
 'উত্তর',
 'উনি',
 'উপর',
 'উপরে',
 'এ',
 'এঁদের',
 'এঁরা',
 'এই',
 'একই',
 'একটি',
 'একবার',
 'একে',
 'এক্',
 'এখন',
 'এখনও',
 'এখানে',
 'এখানেই',
 'এটা',
 'এটাই',
 'এটি',
 'এত',
 'এতটাই',
 'এতে',
 'এদের',
 'এব',
 'এবং',
 'এবার',
 'এমন',
 'এমনকী',
 'এমনি',
 'এর',
 'এরা',
 'এল',
 'এস',
 'এসে',
 'ঐ',
 'ও',
 'ওঁদের',
 'ওঁর',
 'ওঁরা',
 'ওই',
 'ওকে',
 'ওখানে',
 'ওদের',
 'ওর',
 'ওরা',
 'কখনও',
 'কত',
 'কবে',
 'কমনে',
 'কয়েক',
 'কয়েকটি',
 'করছে',
 'করছেন',
 'করতে',
 'করবে',
 'করবেন',
 'করলে',
 'করলেন',
 'করা',
 'করাই',
 'করায়',
 'করার',
 'করি',
 'করিতে',
 'করিয়া',
 'করিয়ে',
 'করে',
 'করেই',
 'করেছিলেন',
 'করেছে',
 'করেছেন',
 'করেন',
 'কাউকে',
 'কাছ',
 'কাছে',
 'কাজ',
 'কাজে',
 'কারও',
 'কারণ',
 'কি',
 'কিংবা',
 'কিছু',
 'কিছুই',
 'কিন্তু',
 'কী',
 'কে',
 'কেউ',
 'কেউই',
 'কেখা',
 'কেন',
 'কোটি',
 'কোন',
 'কোনও',
 'কোনো',
 'ক্ষেত্রে',
 'কয়েক',
 'খুব',
 'গিয়ে',
 'গিয়েছে',
 'গিয়ে',
 'গুলি',
 'গেছে',
 'গেল',
 'গেলে',
 'গোটা',
 'চলে',
 'চান',
 'চায়',
 'চার',
 'চালু',
 'চেয়ে',
 'চেষ্টা',
 'ছাড়া',
 'ছাড়াও',
 'ছিল',
 'ছিলেন',
 'জন',
 'জনকে',
 'জনের',
 'জন্য',
 'জন্যওজে',
 'জানতে',
 'জানা',
 'জানানো',
 'জানায়',
 'জানিয়ে',
 'জানিয়েছে',
 'জে',
 'জ্নজন',
 'টি',
 'ঠিক',
 'তখন',
 'তত',
 'তথা',
 'তবু',
 'তবে',
 'তা',
 'তাঁকে',
 'তাঁদের',
 'তাঁর',
 'তাঁরা',
 'তাঁাহারা',
 'তাই',
 'তাও',
 'তাকে',
 'তাতে',
 'তাদের',
 'তার',
 'তারপর',
 'তারা',
 'তারৈ',
 'তাহলে',
 'তাহা',
 'তাহাতে',
 'তাহার',
 'তিনঐ',
 'তিনি',
 'তিনিও',
 'তুমি',
 'তুলে',
 'তেমন',
 'তো',
 'তোমার',
 'থাকবে',
 'থাকবেন',
 'থাকা',
 'থাকায়',
 'থাকে',
 'থাকেন',
 'থেকে',
 'থেকেই',
 'থেকেও',
 'দিকে',
 'দিতে',
 'দিন',
 'দিয়ে',
 'দিয়েছে',
 'দিয়েছেন',
 'দিলেন',
 'দু',
 'দুই',
 'দুটি',
 'দুটো',
 'দেওয়া',
 'দেওয়ার',
 'দেওয়া',
 'দেখতে',
 'দেখা',
 'দেখে',
 'দেন',
 'দেয়',
 'দ্বারা',
 'ধরা',
 'ধরে',
 'ধামার',
 'নতুন',
 'নয়',
 'না',
 'নাই',
 'নাকি',
 'নাগাদ',
 'নানা',
 'নিজে',
 'নিজেই',
 'নিজেদের',
 'নিজের',
 'নিতে',
 'নিয়ে',
 'নিয়ে',
 'নেই',
 'নেওয়া',
 'নেওয়ার',
 'নেওয়া',
 'নয়',
 'পক্ষে',
 'পর',
 'পরে',
 'পরেই',
 'পরেও',
 'পর্যন্ত',
 'পাওয়া',
 'পাচ',
 'পারি',
 'পারে',
 'পারেন',
 'পি',
 'পেয়ে',
 'পেয়্র্',
 'প্রতি',
 'প্রভৃতি',
 'প্রযন্ত',
 'প্রায়',
 'প্রায়',
 'ফলে',
 'ফিরে',
 'ফের',
 'বক্তব্য',
 'বদলে',
 'বন',
 'বরং',
 'বলতে',
 'বলল',
 'বললেন',
 'বলা',
 'বলে',
 'বলেছেন',
 'বলেন',
 'বসে',
 'বহু',
 'বা',
 'বাদে',
 'বার',
 'বি',
 'বিনা',
 'বিভিন্ন',
 'বিশেষ',
 'বিষয়টি',
 'বেশ',
 'বেশি',
 'ব্যবহার',
 'ব্যাপারে',
 'ভাবে',
 'ভাবেই',
 'মতো',
 'মতোই',
 'মধ্যভাগে',
 'মধ্যে',
 'মধ্যেই',
 'মধ্যেও',
 'মনে',
 'মাত্র',
 'মাধ্যমে',
 'মোট',
 'মোটেই',
 'যখন',
 'যত',
 'যতটা',
 'যথেষ্ট',
 'যদি',
 'যদিও',
 'যা',
 'যাঁর',
 'যাঁরা',
 'যাওয়া',
 'যাওয়ার',
 'যাওয়া',
 'যাকে',
 'যাচ্ছে',
 'যাতে',
 'যাদের',
 'যান',
 'যাবে',
 'যায়',
 'যার',
 'যারা',
 'যিনি',
 'যে',
 'যেখানে',
 'যেতে',
 'যেন',
 'যেমন',
 'র',
 'রকম',
 'রয়েছে',
 'রাখা',
 'রেখে',
 'শুধু',
 'শুরু',
 'সঙ্গে',
 'সঙ্গেও',
 'সব',
 'সবার',
 'সমস্ত',
 'সহ',
 'সহিত',
 'সাধারণ',
 'সামনে',
 'সি',
 'সুতরাং',
 'সে',
 'সেই',
 'সেখান',
 'সেখানে',
 'সেটা',
 'সেটাই',
 'সেটাও',
 'সেটি',
 'স্পষ্ট',
 'স্বয়ং',
 'হইতে',
 'হইবে',
 'হইয়া',
 'হওয়া',
 'হওয়ায়',
 'হওয়ার',
 'হচ্ছে',
 'হত',
 'হতে',
 'হতেই',
 'হন',
 'হবে',
 'হবেন',
 'হয়',
 'হয়তো',
 'হয়নি',
 'হয়ে',
 'হয়েই',
 'হয়েছিল',
 'হয়েছে',
 'হয়েছেন',
 'হল',
 'হলে',
 'হলেই',
 'হলেও',
 'হলো',
 'হাজার',
 'হিসাবে',
 'হৈলে',
 'হোক',
 'হয়']
import pickle
app = Flask(__name__)

model=pickle.load(open('model.pkl','rb'))

from flask_mysqldb import MySQL
app.config['MYSQL_HOST']='localhost'
app.config['MYSQL_USER']='root'
app.config['MYSQL_PASSWORD']=''
app.config['MYSQL_DB']='tfidf_vector'

mysql= MySQL(app)
''''
k=0
def returnpair(t_clean):
 global k
 final = []
 #inner_list=[]
 for k in range(len(t_clean)):
  cur = mysql.connection.cursor()
  sval = t_clean[k]
  cur.execute("SELECT * FROM score_tfidf where indexone='" + sval + "' ")
  p = cur.fetchone()
  newlist=list(p)
 for i in newlist:
   inner_list = []
   inner_list.append([sval,i])
   final.append(inner_list)
 return newlist
'''''
i=0

def returnval(t_clean):
 global i
 final = []
 for i in range(len(t_clean)):
  cur = mysql.connection.cursor()
  sval = t_clean[i]
  cur.execute("SELECT * FROM third_table where indexone='" + sval + "' ")
  p = cur.fetchone()
  newlist=list(p)
  final.append(newlist)
 return final


@app.route("/home", methods=["POST", "GET"])
def home():
 try:
    if request.method=="POST":
        val=request.form["keyword"]
        pnct = '!"#$%&\'()*+,-./:;<=>?‘’@[\\]^_`{|}~।'
        txt_nopunctuation = "".join([c for c in val if c not in pnct]) #punctuation removed
        nltk_tokens = nltk.word_tokenize(txt_nopunctuation)   #tokenization
        txt_clean = [word for word in nltk_tokens if word not in stopwords]  #stop word removal


        #print("This is the list ", returnpair(txt_clean))
        doc_list = []
        list_list=returnval(txt_clean)
        #print("hello",list_list)
        for i in range(len(list_list)):
         for j in range(len(list_list[i])):
          tfidf_val=list_list[i][j]
          if tfidf_val!=0 and j>0:
           #print(j-1,"\n")
           doc_list.append([j-1,list_list[i][0]])
           #print(doc_list,"\n") #index of documents and token pair
        print("Keyword-Document pair retieved from database ")
        print(doc_list)
        print("\n")

        pred_input=[]

        for i in range(0,len(doc_list)):
         for j in range(0,len(doc_list[i])):
          a=doc_list[i][1]
          label=dict_mapping[doc_list[i][1]]
         pred_input.append([doc_list[i][0],label])

        print("Model input for prediction")
        print(pred_input)
        print("\n")


        doc_list=[1,2,3]
        prediction=model.predict(pred_input)
        print("Predicted bid of corresponding document")
        print(prediction.tolist())
        print("\n")

        for i in range(0, len(pred_input)):
         pred_input[i].append(prediction[i])
        print("Docid, Keyword, Bid")
        print(pred_input)
        print("\n")

        pred_input = sorted(pred_input, key=lambda x: x[2], reverse=True)
        print("Sorted Result")
        print(pred_input)
        print("\n")

        final_doc = []
        for i in pred_input:
         final_doc.append(i[0])
        print("Final Document")
        print(final_doc)
        print("\n")
        #remove similar documents and keep nonrepeating documents

        n = len(final_doc)
        mp = dict()
        arr = final_doc

        for i in range(n):
         if arr[i] in mp.keys():
          mp[arr[i]] += 1
         else:
          mp[arr[i]] = 1

        nonredundant_doc = []

        for x in mp:
         nonredundant_doc.append(x)
        print("Non redundant doc")
        print(nonredundant_doc)
        print("\n")

        '''''
        n = len(doc_list)
        mp = dict()
        arr=doc_list

        for i in range(n):
         if arr[i] in mp.keys():
          mp[arr[i]] += 1
         else:
          mp[arr[i]] = 1

        nonredundant_doc=[]

        for x in mp:
         nonredundant_doc.append(x)
        #print(nonredundant_doc)
        '''

        import pandas as pd
        data = pd.read_csv('F:\mydemo.csv')
        l=data.to_numpy()
        pass_list=[]
        for i in nonredundant_doc:
         #print(i,"\n")
         pass_list.append(l[i])

        #for i in pass_list:
         #print(i[0])

        a = "\""+ val + "\" এর জন্য ফলাফল প্রদর্শিত হচ্ছে !"




        #return returnval(txt_clean)
        #return redirect(url_for('result', key_val=l))
        #return f"<h1>{returnval(txt_clean)}</h1>"
        return render_template("index.html",pass_list=pass_list,a=a)

    else:
        return render_template('home.html')
 except:
  a="No Result for "+val+"!!"
  return render_template("index.html", a=a)
@app.route("/<key_val>")
def result(key_val):
    return f"<h1>{key_val}</h1>"


#not necessary
@app.route('/')
def index():
 cur=mysql.connection.cursor()
 #cur.execute("INSERT INTO book_details(book_id,title,price) VALUES (%s, %s, %s)",(bid,bname,bprice))
 cur.execute("SELECT * FROM score_tfidf where indexone='1' ")
 #cur.execute("SELECT * FROM book_details where indexone=4 ")
 p=cur.fetchone()
 print(p)
 mysql.connection.commit()
 cur.close()
 predictor=model.predict([[65,	419],[65,	12],[0,11020],])
 return f"<h1>{predictor}</h1>"



