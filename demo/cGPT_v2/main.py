import pickle
from time import sleep

from chatgpt_wrapper import ChatGPT
import json
from datetime import datetime
import  utils

################
### some prompts
"""
属性词指评价的对象，情感倾向是指对象在上下文中的整体情感，从”正向”，"中性”，“负向三者中选择，观点词是评论对象时用到的描述词，属性词和观点词如果不存在就标记为无。如针对“这盘空心菜很好吃”，输出为“属性词：空心菜；情感倾向：正向；观点词：很好吃”；针对“这个餐厅的食物很好吃，但环境不好”，输出为“属性词：食物；情感倾向：正向；观点词：好吃；属性词：环境；情感倾向：负向；观点词：不好”。极性不能是null，观点词和方面可以是null。请针对请输出下面几句话的属性词、对应的情感倾向和对应的观点词:
1、这盘空心菜很好吃
2、这个游戏不好玩
3、小溪的水好脏
4、这个石头很重，我抬不动
5、这个车的底盘很重
6、这家餐厅环境不好，但是食物美味
7、"两个月前感冒高烧嗓子疼干咳，这次感冒低烧嗓子疼流鼻涕，本命年不能更sui了，就没有什么顺心如意的事，赶紧过去吧。?"
"""
"""
"Aspect term  is the opinion target which explicitly appears in the given text, e.g., “pizza” in the sentence“The pizza is delicious.” When the target is implicitly expressed (e.g., “It is overpriced!”), we denote the aspect term as a special one named “null“. Opinion term  is the expression given by the opinion holder to express his/her sentiment towards the target. For instance, “delicious” is the opinion term in the running example “The pizza is delicious”. Sentiment polarity  describes the orientation of the sentiment over an aspect category or an aspect term, which usually includes \"positive\", \" negative\", and \"neutral\". Please give me the aspects, opinions and sentiment polarities in the following sentences with the type of \"aspect:;opinion:;polarity:\". "
"""
"""
"Aspect category  defines a unique aspect of an entity and is supposed to be one in  \" \'RESTAURANT#GENERAL\', \'SERVICE#GENERAL\', \'FOOD#GENERAL\', \'FOOD#QUALITY\', \'FOOD#STYLE_OPTIONS\', \'DRINKS#STYLE_OPTIONS\', \'DRINKS#PRICES\',\'AMBIENCE#GENERAL\', \'RESTAURANT#PRICES\', \'FOOD#PRICES\', \'RESTAURANT#MISCELLANEOUS\', \'DRINKS#QUALITY\', \'LOCATION#GENERAL\'   \". For example, food and service can be aspect categories for the restaurant domain. Aspect term  is the opinion target which explicitly appears in the given text, e.g., “pizza” in the sentence“The pizza is delicious.” When the target is implicitly expressed (e.g., “It is overpriced!”), we denote the aspect term as a special one named “null“. Opinion term  is the expression given by the opinion holder to express his/her sentiment towards the target. For instance, “delicious” is the opinion term in the running example “The pizza is delicious”. Sentiment polarity  describes the orientation of the sentiment over an aspect category or an aspect term, which usually includes \"positive\", \" negative\", and \"neutral\". Please give me the aspects and its category, opinions and sentiment polarities in the following sentences with the type of \"aspect:;aspect category:;opinion:;polarity:;\".0. The food is great but the price is too high. "
"""
#################
#########
# debug setting
########
type=4

###############
f = open('./data/data/test_pair.pkl', "rb")
triple_data = pickle.load(f)  # 格式： [(aspect_index,opinion_index,sentiment)]
f.close()
print(0)
bot = ChatGPT()
if type==3:
    response = bot.ask("Aspect term  is the opinion target which explicitly appears in the given text, e.g., “pizza” in the sentence“The pizza is delicious.” When the target is implicitly expressed (e.g., “It is overpriced!”), we denote the aspect term as a special one named “null“. Opinion term  is the expression given by the opinion holder to express his/her sentiment towards the target. For instance, “delicious” is the opinion term in the running example “The pizza is delicious”. Sentiment polarity  describes the orientation of the sentiment over an aspect category or an aspect term, which usually includes \"positive\", \" negative\", and \"neutral\". Please give me the aspects, opinions and sentiment polarities in the following sentences with the type of \"aspect:;opinion:;polarity:\". ")
    print(response)  # prints the response from chatGPT
    file_read = open('./data/data/14laptest.json', 'r', encoding='utf-8')
    file_content = json.load(file_read)
    file_read.close()
    print(1)
    dt = datetime.now()
    t=dt.strftime('%Y-%m-%d-%H-%M-%S-%f')
    logger = utils.get_logger("./log", time=dt.strftime('%Y-%m-%d-%H-%M-%S'))

    logger, fh, sh = utils.get_logger("./log")
    logger.info('start testing......')
    # load checkpoint
    logger.info('loading checkpoint......')
    with open('./answer'+t+'.json', 'w+', encoding='utf-8') as file:
        re=[]
        for n,item in enumerate(file_content):
            dic={}
            # if n>2:
            #     break
            sleep(0.35)
            text=item['text']
            st=str(n+1)+'.'
            for ci in text:
                st+=ci+' '
            response=bot.ask(st)
            dic['response']=response
            dic['text']=st
            dic['id']=n
            answer=[]
            for jtem in triple_data[n]:
                aspects=jtem[0]
                opinions=jtem[1]
                polarity=jtem[2]
                aspect=''
                opinion=''
                if len(aspects)==2:
                    for ktem in range(aspects[0],aspects[1]+1):
                        aspect+=text[ktem]+' '
                else:
                    aspect=text[aspects[0]]

                if len(opinions) == 2:
                    for ktem in range(opinions[0], opinions[1] + 1):
                        opinion += text[ktem] + ' '
                else:
                    opinion = text[opinions[0]]

                p=''
                if polarity==0:
                    p='neural'
                elif polarity==1:
                    p='positive'
                else:
                    p='negative'
                answer.append((aspect,opinion,polarity))
            dic['aspect']=answer
            logger.info(st)
            logger.info(response)
            re.append(dic)
        json.dump(re, file,indent=6)
elif type==4:
    response = bot.ask("Aspect category  defines a unique aspect of an entity and is supposed to be one in  \" \'RESTAURANT#GENERAL\', \'SERVICE#GENERAL\', \'FOOD#GENERAL\', \'FOOD#QUALITY\', \'FOOD#STYLE_OPTIONS\', \'DRINKS#STYLE_OPTIONS\', \'DRINKS#PRICES\',\'AMBIENCE#GENERAL\', \'RESTAURANT#PRICES\', \'FOOD#PRICES\', \'RESTAURANT#MISCELLANEOUS\', \'DRINKS#QUALITY\', \'LOCATION#GENERAL\'   \". For example, food and service can be aspect categories for the restaurant domain. Aspect term  is the opinion target which explicitly appears in the given text, e.g., “pizza” in the sentence“The pizza is delicious.” When the target is implicitly expressed (e.g., “It is overpriced!”), we denote the aspect term as a special one named “null“. Opinion term  is the expression given by the opinion holder to express his/her sentiment towards the target. For instance, “delicious” is the opinion term in the running example “The pizza is delicious”. Sentiment polarity  describes the orientation of the sentiment over an aspect category or an aspect term, which usually includes \"positive\", \" negative\", and \"neutral\". Please give me the aspects and its category, opinions and sentiment polarities in the following sentences with the type of \"aspect:;aspect category:;opinion:;polarity:;\".0. The food is great but the price is too high. ")
    print(response)  # prints the response from chatGPT
    file_read = open('./data/data/rest16test.json', 'r', encoding='utf-8')
    file_content = json.load(file_read)
    file_read.close()
    print(1)
    dt = datetime.now()
    t=dt.strftime('%Y-%m-%d-%H-%M-%S-%f')
    logger = utils.get_logger("./log", time=dt.strftime('%Y-%m-%d-%H-%M-%S'))

    logger, fh, sh = utils.get_logger("./log")
    logger.info('start testing......')
    # load checkpoint
    with open('./answer'+t+'.json', 'w+', encoding='utf-8') as file:
        re=[]
        stt=''
        for n,item in enumerate(file_content):
            print('****')
            if n%4!=0:
                text = item['text']
                st = str(n + 1) + '. '
                for ci in text:
                    st += ci + ' '
`               stt=stt+st
            response=bot.ask(stt)
            # dic['response']=response
            # dic['text']=stt
            # dic['id']=n
            # answer=[]
            # quad=item['quad']
            # """
            # "quad": [{"aspect_index": "-1,-1", "category": "FOOD#QUALITY", "polarity": "2", "opinion_index": "0,1", "aspect": [], "opinion": ["yum"]}]
            # """
            # for jtem in quad:
            #     aspect=jtem['aspect']
            #     category=jtem['category']
            #     opinion=jtem['opinion']
            #     polarity=jtem['polarity']
            #     answer.append((aspect,category,opinion,polarity))
            # dic['aspect']=answer
            logger.info(stt)
            logger.info(response)
            re.append(dic)
            stt=''
        json.dump(re, file,indent=6)
