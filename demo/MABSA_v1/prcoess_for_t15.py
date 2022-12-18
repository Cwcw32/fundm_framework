import json
import os

def process(datapath):
    dev=datapath+'/dev.txt'
    test=datapath+'/test.txt'
    train=datapath+'/train.txt'
    fin=open(dev,'r',encoding='utf-8')
    dev_content=fin.readlines()
    fin.close()
    fin=open(test,'r',encoding='utf-8')
    test_content=fin.readlines()
    fin.close()
    fin=open(train,'r',encoding='utf-8')
    train_content=fin.readlines()
    fin.close()
    a=[dev_content,test_content,train_content]
    b=['dev','test','train']
    for n,content in enumerate(a):
        c=[]
        with open('./dataset/data/T15/data/'+b[n]+'.json','w',encoding='utf-8') as file:
            for i in range(0,len(content),4):
                d={}
                text_left,_,text_right=[s.strip() for s in content[i].partition("$T$")]
                d['aspect']=content[i+1].strip()
                d['text']=text_left+' '+ d['aspect']+ ' ' + text_right
                d['emotion_label']=int(content[i+2].strip())+1
                d['id']=content[i+3].strip().replace('.jpg','')
                c.append(d)
            json.dump(c,file,ensure_ascii=False,indent=4)
                #file.write('\n')



"""
    RT @ ltsChuckBass: $T$ is everything  # MCM
    Chuck Bass
    1
    1860693.jpg
"""

if __name__ =='__main__':
    process('./dataset/data/T15')