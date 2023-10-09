import random
import numpy as np
import math
import turtle as t

random.seed(10)

#----------------------定义一个类，实现ID3，CART,C4.5决策树，类可以传递两个参数，一个是数据集的路径，另一个是要使用的决策树类型（ID3\C4.5\CART）-----------------#
class Decision_Tree:
    ##################################初始化类，传入数据集参数和模式##############################
    def __init__(self,DataSet_Path=r'./watermelon2.0.csv',Mode='ID3'):
        self.DataSet_Path=DataSet_Path
        self.Mode=Mode
    ################################对传入的数据进行处理#######################################
    def Create_DataSet(self):

        self.DataSet=open(self.DataSet_Path).read()
        self.DataSet=self.DataSet.split('\n')
        for i in range(len(self.DataSet)):
            self.DataSet[i]=self.DataSet[i].split(',')
        self.DataSet=np.array(self.DataSet[0:len(self.DataSet)-1])
        self._TrainDataset=self.DataSet.tolist()
        self._TestDataSet=[]
        for i in sorted(random.sample((np.where(self.DataSet[:,6]=='是')[0]).tolist(),2)+random.sample(list(np.where(self.DataSet[:,6]=='否')[0].tolist()),2),reverse=True):
            self._TestDataSet.append(self.DataSet[i].tolist())
            self._TrainDataset.pop(i)

        return self._TrainDataset,self._TestDataSet,self.DataSet.tolist()

    def ID3_Tree(self,_TrainDataset=None,_Parameter=None):  #属性划分方法采用Info_Gain方法

        if _TrainDataset==None:
            _TrainDataset=self._TrainDataset
        if _Parameter==None:
            _Parameter=[]


        NumYes=list(np.array(_TrainDataset)[:, len(_TrainDataset[0])-1]).count('是')
        NumNo=list(np.array(_TrainDataset)[:, len(_TrainDataset[0])-1]).count('否')
        NumAll=NumYes+NumNo
        if NumYes == 0:
            EntD = -(+ NumNo / NumAll * math.log((NumNo / NumAll), 2))
        elif NumNo == 0:
            EntD = -(NumYes / NumAll * math.log((NumYes / NumAll), 2))
        else:
            EntD = -(NumYes / NumAll * math.log((NumYes / NumAll), 2) + NumNo / NumAll * math.log((NumNo / NumAll), 2))
        #EntD=-(NumYes/NumAll*math.log((NumYes/NumAll),2)+NumNo/NumAll*math.log((NumNo/NumAll),2))
        Gain = []
        for i in range(0,len(_TrainDataset[0])-1):
            count=0
            for item in sorted(set(np.array(_TrainDataset[1:len(_TrainDataset)])[:, i])):
                subLabels = []
                for value in np.where((np.array(_TrainDataset[1:len(_TrainDataset)])[:,i])==item)[0]:
                    subLabels.append(np.array(_TrainDataset[1:len(_TrainDataset)])[:,len(_TrainDataset[0])-1].tolist()[value])
                NumYes=subLabels.count('是')
                NumNo=subLabels.count('否')
                NumAll=NumYes+NumNo
                if NumYes==0:
                    Ent = -(+ NumNo / NumAll * math.log((NumNo / NumAll), 2))
                elif NumNo==0:
                    Ent=-(NumYes / NumAll * math.log((NumYes / NumAll), 2) )
                else:
                    Ent=-(NumYes/NumAll*math.log((NumYes/NumAll),2)+NumNo/NumAll*math.log((NumNo/NumAll),2))

                count=count+Ent*list(np.array(_TrainDataset)[:, i]).count(item)/(len(list(np.array(_TrainDataset)[:, i]))-1)
            Gain.append(EntD-count)              ##得到信息增益值





        for i in sorted(set(np.array(_TrainDataset[1:len(_TrainDataset)])[:, Gain.index(max(Gain))])):
            Atribute=np.array(_TrainDataset[0]).tolist()
            del Atribute[Gain.index(max(Gain))]

            _TrainDataView=[]
            for value in np.where(np.array(_TrainDataset[1:len(_TrainDataset)])[:, Gain.index(max(Gain))]==i)[0].tolist():
                elseL=np.array(_TrainDataset[1:len(_TrainDataset)])[value].tolist()
                del elseL[Gain.index(max(Gain))]
                _TrainDataView.append(elseL)
            _TrainDataView=[Atribute]+_TrainDataView
            #print(_TrainDataView)
            if np.array(_TrainDataView[1:len(_TrainDataView)])[:,-1].tolist().count(np.array(_TrainDataView[1:len(_TrainDataView)])[:,-1].tolist()[0])==len(_TrainDataView[1:len(_TrainDataView)]) or len(_TrainDataView)==2:
                F_Parameter = []
                F_Parameter.append(np.array(_TrainDataset[0]).tolist()[Gain.index(max(Gain))])
                F_Parameter.append(i)
                F_Parameter.append(np.array(_TrainDataView[1:len(_TrainDataView)])[:,-1].tolist()[0])
                _Parameter.append(F_Parameter)


            else:
                F_Parameter = []
                F_Parameter.append(np.array(_TrainDataset[0]).tolist()[Gain.index(max(Gain))])
                F_Parameter.append(i)
                _Parameter.append(F_Parameter)
                Decision_Tree.ID3_Tree(self, _TrainDataView, _Parameter=_Parameter)

        return _Parameter



    def C4Point5_Tree(self,_TrainDataset=None,_Parameter=None): #属性划分方法采用Info_Gain_radio方法
        if _TrainDataset==None:
            _TrainDataset=self._TrainDataset
        if _Parameter==None:
            _Parameter=[]


        NumYes=list(np.array(_TrainDataset)[:, len(_TrainDataset[0])-1]).count('是')
        NumNo=list(np.array(_TrainDataset)[:, len(_TrainDataset[0])-1]).count('否')
        NumAll=NumYes+NumNo
        if NumYes == 0:
            EntD = -(+ NumNo / NumAll * math.log((NumNo / NumAll), 2))
        elif NumNo == 0:
            EntD = -(NumYes / NumAll * math.log((NumYes / NumAll), 2))
        else:
            EntD = -(NumYes / NumAll * math.log((NumYes / NumAll), 2) + NumNo / NumAll * math.log((NumNo / NumAll), 2))
        #EntD=-(NumYes/NumAll*math.log((NumYes/NumAll),2)+NumNo/NumAll*math.log((NumNo/NumAll),2))

        Gain = []

        for i in range(0,len(_TrainDataset[0])-1):
            count=0
            IV=0
            for item in sorted(set(np.array(_TrainDataset[1:len(_TrainDataset)])[:, i])):
                subLabels = []
                for value in np.where((np.array(_TrainDataset[1:len(_TrainDataset)])[:,i])==item)[0]:
                    subLabels.append(np.array(_TrainDataset[1:len(_TrainDataset)])[:,len(_TrainDataset[0])-1].tolist()[value])
                NumYes=subLabels.count('是')
                NumNo=subLabels.count('否')
                NumAll=NumYes+NumNo
                if NumYes==0:
                    Ent = -(+ NumNo / NumAll * math.log((NumNo / NumAll), 2))
                elif NumNo==0:
                    Ent=-(NumYes / NumAll * math.log((NumYes / NumAll), 2) )
                else:
                    Ent=-(NumYes/NumAll*math.log((NumYes/NumAll),2)+NumNo/NumAll*math.log((NumNo/NumAll),2))

                count=count+Ent*list(np.array(_TrainDataset)[:, i]).count(item)/(len(list(np.array(_TrainDataset)[:, i]))-1)
                IV=IV+(np.array(_TrainDataset[1:len(_TrainDataset)])[:, i].tolist().count(item))/len(np.array(_TrainDataset[1:len(_TrainDataset)])[:,i])

            Gain.append((EntD-count)/IV)              ##得到信息增益率





        for i in sorted(set(np.array(_TrainDataset[1:len(_TrainDataset)])[:, Gain.index(max(Gain))])):
            Atribute=np.array(_TrainDataset[0]).tolist()
            del Atribute[Gain.index(max(Gain))]

            _TrainDataView=[]
            for value in np.where(np.array(_TrainDataset[1:len(_TrainDataset)])[:, Gain.index(max(Gain))]==i)[0].tolist():
                elseL=np.array(_TrainDataset[1:len(_TrainDataset)])[value].tolist()
                del elseL[Gain.index(max(Gain))]
                _TrainDataView.append(elseL)
            _TrainDataView=[Atribute]+_TrainDataView
            #print(_TrainDataView)
            if np.array(_TrainDataView[1:len(_TrainDataView)])[:,-1].tolist().count(np.array(_TrainDataView[1:len(_TrainDataView)])[:,-1].tolist()[0])==len(_TrainDataView[1:len(_TrainDataView)]) or len(_TrainDataView)==2:
                F_Parameter = []
                F_Parameter.append(np.array(_TrainDataset[0]).tolist()[Gain.index(max(Gain))])
                F_Parameter.append(i)
                F_Parameter.append(np.array(_TrainDataView[1:len(_TrainDataView)])[:,-1].tolist()[0])
                _Parameter.append(F_Parameter)


            else:
                F_Parameter = []
                F_Parameter.append(np.array(_TrainDataset[0]).tolist()[Gain.index(max(Gain))])
                F_Parameter.append(i)
                _Parameter.append(F_Parameter)
                Decision_Tree.ID3_Tree(self, _TrainDataView, _Parameter=_Parameter)

        return _Parameter

    def CART_Tree(self,_TrainDataset=None,_Parameter=None):     #属性划分方法采用gini方法
        if _TrainDataset==None:
            _TrainDataset=self._TrainDataset
        if _Parameter==None:
            _Parameter=[]


        NumYes=list(np.array(_TrainDataset)[:, len(_TrainDataset[0])-1]).count('是')
        NumNo=list(np.array(_TrainDataset)[:, len(_TrainDataset[0])-1]).count('否')
        NumAll=NumYes+NumNo
        if NumYes == 0:
            EntD = ( NumNo / NumAll * (1-(NumNo / NumAll)))
        elif NumNo == 0:
            EntD = (NumYes / NumAll * (1-(NumYes / NumAll)))
        else:
            EntD = (NumYes / NumAll * (1-(NumYes / NumAll) ) + NumNo / NumAll * (1-(NumNo / NumAll)))
        #EntD=-(NumYes/NumAll*math.log((NumYes/NumAll),2)+NumNo/NumAll*math.log((NumNo/NumAll),2))
        Gain = []
        for i in range(0,len(_TrainDataset[0])-1):
            count=0
            for item in sorted(set(np.array(_TrainDataset[1:len(_TrainDataset)])[:, i])):
                subLabels = []
                for value in np.where((np.array(_TrainDataset[1:len(_TrainDataset)])[:,i])==item)[0]:
                    subLabels.append(np.array(_TrainDataset[1:len(_TrainDataset)])[:,len(_TrainDataset[0])-1].tolist()[value])
                NumYes=subLabels.count('是')
                NumNo=subLabels.count('否')
                NumAll=NumYes+NumNo
                if NumYes==0:
                    Ent = -(+ NumNo / NumAll * (1-(NumNo / NumAll)))
                elif NumNo==0:
                    Ent=-(NumYes / NumAll * (1-(NumYes / NumAll)) )
                else:
                    Ent=-(NumYes/NumAll*(1-(NumYes/NumAll))+NumNo/NumAll*(1-(NumNo/NumAll)))

                count=count+Ent*list(np.array(_TrainDataset)[:, i]).count(item)/(len(list(np.array(_TrainDataset)[:, i]))-1)
            Gain.append(EntD-count)              ##得到信息增益值





        for i in sorted(set(np.array(_TrainDataset[1:len(_TrainDataset)])[:, Gain.index(max(Gain))])):
            Atribute=np.array(_TrainDataset[0]).tolist()
            del Atribute[Gain.index(max(Gain))]

            _TrainDataView=[]
            for value in np.where(np.array(_TrainDataset[1:len(_TrainDataset)])[:, Gain.index(max(Gain))]==i)[0].tolist():
                elseL=np.array(_TrainDataset[1:len(_TrainDataset)])[value].tolist()
                del elseL[Gain.index(max(Gain))]
                _TrainDataView.append(elseL)
            _TrainDataView=[Atribute]+_TrainDataView
            #print(_TrainDataView)
            if np.array(_TrainDataView[1:len(_TrainDataView)])[:,-1].tolist().count(np.array(_TrainDataView[1:len(_TrainDataView)])[:,-1].tolist()[0])==len(_TrainDataView[1:len(_TrainDataView)]) or len(_TrainDataView)==2:
                F_Parameter = []
                F_Parameter.append(np.array(_TrainDataset[0]).tolist()[Gain.index(max(Gain))])
                F_Parameter.append(i)
                F_Parameter.append(np.array(_TrainDataView[1:len(_TrainDataView)])[:,-1].tolist()[0])
                _Parameter.append(F_Parameter)


            else:
                F_Parameter = []
                F_Parameter.append(np.array(_TrainDataset[0]).tolist()[Gain.index(max(Gain))])
                F_Parameter.append(i)
                _Parameter.append(F_Parameter)
                Decision_Tree.ID3_Tree(self, _TrainDataView, _Parameter=_Parameter)

        return _Parameter


    def Drawing(self):
        t.screensize(800, 600)
        t.pensize(2)  # 设置画笔的大小
        t.speed(10)  # 设置画笔速度为10
        t.write("这个图画起来比较繁琐，\n在网上找了一些教程，\n都是利用sklearn进行画图，\n当然也有以标准输入输出画图的，但是\n显然我的输出\n不是标准决策树输出，所以打开画图软件，画图\n", align="center", font=("楷体", 16, "bold"))
        t.end_fill()
        t.done()

    def Packing_Department(self,_TestDataSet=None):
        if _TestDataSet==None:
            _TestDataSet=self._TestDataSet

        if self.Mode=='ID3':
            Attribute_Parameter=Decision_Tree.ID3_Tree(self)
            Attribute_Parameter=sum(Attribute_Parameter,[])
            StarCar,Star =[],[]
            Parmeter=Attribute_Parameter
            for i in  Parmeter:

                if i != '是' and i != '否':
                    StarCar.append(i)
                    if StarCar[len(StarCar) - 2] == '是' or StarCar[len(StarCar) - 2] == '否':
                        for value in range(np.where(np.array(StarCar) == StarCar[len(StarCar) - 1])[0].tolist()[1] -np.where(np.array(StarCar) == StarCar[len(StarCar) - 1])[0].tolist()[0]):
                            del StarCar[-1]
                else:
                    StarCar.append(i)
                    Star=Star+StarCar
            Star_w,Star,Flash,_Prediect,ReTestDataCount,UseTest=Star,[],[],[],[],[]

            for item in Star_w:
                if item!='是' and item!='否':
                    Flash.append(item)
                else:
                    Flash.append(item)
                    Star.append(Flash)
                    Flash=[]
            for item in _TestDataSet:
                ReTestDataCount=[]
                for value in range(len(item)):
                    ReTestDataCount.append(self._TrainDataset[0][value])
                    ReTestDataCount.append(item[value])
                UseTest.append(ReTestDataCount)

            for value in UseTest:
                for i in Star:
                    count=0
                    for item in value[0:len(value)-2]:
                        if item in i:
                            count=count+1
                    if count==len(i)-1:
                        _Prediect.append(i[-1])
            _label=np.array(_TestDataSet)[:,-1].tolist()

            TP, FN, FP, TN =0, 0, 0, 0
            for i in range(len(_Prediect)):
                if _label[i] == '是':
                    if _Prediect[i] == '是':
                        TP = TP + 1  # 计算TP的值
                    else:
                        FN = FN + 1  # 计算FN的值
                else:
                    if _Prediect[i] == '否':
                        TN = TN + 1  # 计算TN的值
                    else:
                        FP = FP + 1  # 计算FP的值
            accuracy = (TP + TN) / (TP + TN + FN + FP)  # 计算准确率
            Precision = TP / (TP + FP)  # 计算查准率
            Recall = TP / (TP + FN)  # 计算查全率
            print("_label={} _prediect={}".format(_label,_Prediect))
            print("TP={} TN={} FP={} FN={} ".format(TP, TN, FP, FN))
            print("accuracy={} Precision={} Recall={}".format(accuracy, Precision, Recall))

            Decision_Tree.Drawing(self)
        elif self.Mode=='C4.5':
            Attribute_Parameter = Decision_Tree.C4Point5_Tree(self)
            Attribute_Parameter = sum(Attribute_Parameter, [])
            StarCar, Star = [], []

            Parmeter = Attribute_Parameter
            for i in Parmeter:

                if i != '是' and i != '否':
                    StarCar.append(i)
                    if StarCar[len(StarCar) - 2] == '是' or StarCar[len(StarCar) - 2] == '否':
                        for value in range(np.where(np.array(StarCar) == StarCar[len(StarCar) - 1])[0].tolist()[1] -
                                           np.where(np.array(StarCar) == StarCar[len(StarCar) - 1])[0].tolist()[0]):
                            del StarCar[-1]
                else:
                    StarCar.append(i)
                    Star = Star + StarCar
            Star_w, Star, Flash, _Prediect, ReTestDataCount, UseTest = Star, [], [], [], [], []

            for item in Star_w:
                if item != '是' and item != '否':
                    Flash.append(item)
                else:
                    Flash.append(item)
                    Star.append(Flash)
                    Flash = []
            for item in _TestDataSet:
                ReTestDataCount = []
                for value in range(len(item)):
                    ReTestDataCount.append(self._TrainDataset[0][value])
                    ReTestDataCount.append(item[value])
                UseTest.append(ReTestDataCount)

            for value in UseTest:
                for i in Star:
                    count = 0
                    for item in value[0:len(value) - 2]:
                        if item in i:
                            count = count + 1
                    if count == len(i) - 1:
                        _Prediect.append(i[-1])
            _label = np.array(_TestDataSet)[:, -1].tolist()

            TP, FN, FP, TN = 0, 0, 0, 0
            for i in range(len(_Prediect)):
                if _label[i] == '是':
                    if _Prediect[i] == '是':
                        TP = TP + 1  # 计算TP的值
                    else:
                        FN = FN + 1  # 计算FN的值
                else:
                    if _Prediect[i] == '否':
                        TN = TN + 1  # 计算TN的值
                    else:
                        FP = FP + 1  # 计算FP的值
            accuracy = (TP + TN) / (TP + TN + FN + FP)  # 计算准确率
            Precision = TP / (TP + FP)  # 计算查准率
            Recall = TP / (TP + FN)  # 计算查全率
            print("_label={} _prediect={}".format(_label, _Prediect))
            print("TP={} TN={} FP={} FN={} ".format(TP, TN, FP, FN))
            print("accuracy={} Precision={} Recall={}".format(accuracy, Precision, Recall))
            Decision_Tree.Drawing(self)
        elif self.Mode=='CART':
            Attribute_Parameter = Decision_Tree.CART_Tree(self)
            Attribute_Parameter = sum(Attribute_Parameter, [])
            StarCar, Star = [], []

            Parmeter = Attribute_Parameter
            for i in Parmeter:

                if i != '是' and i != '否':
                    StarCar.append(i)
                    if StarCar[len(StarCar) - 2] == '是' or StarCar[len(StarCar) - 2] == '否':
                        for value in range(np.where(np.array(StarCar) == StarCar[len(StarCar) - 1])[0].tolist()[1] -
                                           np.where(np.array(StarCar) == StarCar[len(StarCar) - 1])[0].tolist()[0]):
                            del StarCar[-1]
                else:
                    StarCar.append(i)
                    Star = Star + StarCar
            Star_w, Star, Flash, _Prediect, ReTestDataCount, UseTest = Star, [], [], [], [], []

            for item in Star_w:
                if item != '是' and item != '否':
                    Flash.append(item)
                else:
                    Flash.append(item)
                    Star.append(Flash)
                    Flash = []
            for item in _TestDataSet:
                ReTestDataCount = []
                for value in range(len(item)):
                    ReTestDataCount.append(self._TrainDataset[0][value])
                    ReTestDataCount.append(item[value])
                UseTest.append(ReTestDataCount)

            for value in UseTest:
                for i in Star:
                    count = 0
                    for item in value[0:len(value) - 2]:
                        if item in i:
                            count = count + 1
                    if count == len(i) - 1:
                        _Prediect.append(i[-1])
            _label = np.array(_TestDataSet)[:, -1].tolist()

            TP, FN, FP, TN = 0, 0, 0, 0
            for i in range(len(_Prediect)):
                if _label[i] == '是':
                    if _Prediect[i] == '是':
                        TP = TP + 1  # 计算TP的值
                    else:
                        FN = FN + 1  # 计算FN的值
                else:
                    if _Prediect[i] == '否':
                        TN = TN + 1  # 计算TN的值
                    else:
                        FP = FP + 1  # 计算FP的值
            accuracy = (TP + TN) / (TP + TN + FN + FP)  # 计算准确率
            Precision = TP / (TP + FP)  # 计算查准率
            Recall = TP / (TP + FN)  # 计算查全率
            print("_label={} _prediect={}".format(_label, _Prediect))
            print("TP={} TN={} FP={} FN={} ".format(TP, TN, FP, FN))
            print("accuracy={} Precision={} Recall={}".format(accuracy, Precision, Recall))
            Decision_Tree.Drawing(self)


A=Decision_Tree(Mode='ID3')
A.Create_DataSet()
A.Packing_Department()
