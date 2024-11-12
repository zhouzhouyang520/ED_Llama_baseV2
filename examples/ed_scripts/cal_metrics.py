import json
from evaluate import calc_distinct
import datetime
import math
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

def sentiment2label(text):
    emo_set = "surprised,excited,annoyed,proud,angry,sad,grateful,lonely,impressed,afraid,disgusted,confident,terrified,hopeful,anxious,disappointed,joyful,prepared,guilty,furious,nostalgic,jealous,anticipating,embarrassed,content,devastated,sentimental,caring,trusting,ashamed,apprehensive,faithful".lower().split(",")
    #import ipdb; ipdb.set_trace()
    #print(f"emo_set: {len(emo_set)}, {emo_set}")
    #label_dict = {"positive": 2, "neutral": 1, "negative": 0}
    label_dict = dict([(emo, i) for (i, emo) in enumerate(emo_set)])
    #print(f"label_dict: {len(label_dict)}, {label_dict}")

    for key in label_dict: 
        if key in text.lower(): 
            return label_dict[key]
    return 1
 
def read_jsonl_file(file_path):
    res_data = []
    emo_count = 0.0
    emo_total = 0.0
    true_labels = []
    pred_lables = []
    with open(file_path, 'r') as file:
        for line in file:
            #print(f"Line: {line}")
            line_json = json.loads(line)
            emotion = line_json["label"].split("\n ")[0].replace("Emotion:", "").replace(" ", "")
            predict = line_json["predict"].split("\n")
            #print(f"line_json: {line_json}")
            pred_emotion = predict[0].replace("Emotion:", "")
            if len(predict) > 1:
                pred_response = predict[1].replace("Response:", "")  
            else:
                pred_response = "" 
            true_label = sentiment2label(emotion)
            pred_label = sentiment2label(pred_emotion)
            #print(f"emotion: {emotion}, pred_emotion: {pred_emotion}, true_label: {true_label}, pred_label: {pred_label}")
            #print(f"Greedy:{pred_response}")
            #print("===============================")
            true_labels.append(true_label) 
            pred_lables.append(pred_label)
            emo_total += 1.0
            if emotion.lower() in pred_emotion.lower(): 
                emo_count += 1.0 
            res_data.append(pred_response)
    macro_f1 = f1_score(true_labels, pred_lables, average='macro') * 100
    print(classification_report(true_labels, pred_lables))
    return emo_count, emo_total, res_data, macro_f1 

def read_res_dict(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    #print(f"data: {data}")
    return data 
 
def cal_metrics():
    base_dir = "output_ed/"
    file_path = base_dir + 'generated_predictions.jsonl'
    emo_count, emo_total, response_data, macro_f1 = read_jsonl_file(file_path)
    emo_rate = emo_count / emo_total * 100
    print(f"emo_count: {emo_count}, emo_total: {emo_total}, rate: {emo_rate}, macro_f1: {macro_f1}")
    d1, d2  = calc_distinct(response_data)
    print(f"Dist-1: {d1}, Dist-2: {d2}")

#    res_file_path = base_dir + 'all_results.json'
#    res_dict = read_res_dict(res_file_path)
#    if "predict_loss" in res_dict:
#        loss = res_dict["predict_loss"]
#        ppl = math.exp(loss)
#        print(f"loss: {loss}, ppl: {ppl}")
#    else:
    loss, ppl = 0.0, 0.0
    out_path = base_dir + 'ed_result.txt'
    current_time = datetime.datetime.now()
    out_txt = f"{current_time}\tLoss\tPPL\tACC\tDist-1\tDist-2\n{loss}\t{ppl}\t{emo_rate}\t{d1}\t{d2}\n\n"
    with open(out_path, "a") as file: 
        file.write(out_txt) 
 
cal_metrics()
