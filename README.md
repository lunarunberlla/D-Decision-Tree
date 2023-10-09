# Decision Tree

## Introduction:

​			决策树(Decision Tree）是在已知各种情况发生概率的[基础](https://baike.baidu.com/item/基础/32794?fromModule=lemma_inlink)上，通过构成决策树来求取净现值的[期望](https://baike.baidu.com/item/期望/35704?fromModule=lemma_inlink)值大于等于零的概率，评价项目风险，判断其可行性的决策分析方法，是直观运用概率分析的一种图解法。由于这种决策分支画成图形很像一棵树的枝干，故称决策树。在机器学习中，决策树是一个预测模型，他代表的是对象属性与对象值之间的一种映射关系。Entropy = 系统的凌乱程度，使用算法[ID3](https://baike.baidu.com/item/ID3?fromModule=lemma_inlink), [C4.5](https://baike.baidu.com/item/C4.5?fromModule=lemma_inlink)和C5.0生成树算法使用熵。这一度量是基于信息学理论中熵的概念。

## Derivation：

### 		Amount Of Information:

​				信息量在很长的一段时间内，我感觉这是一个很混沌的量，在脑海里对他的印象若即若离，不可琢磨，直到最近我才想到了一个好的方法尝试取理解它，就是在某一个事件发生的概率下，这个事件可能发生的情况的多少。Emmm...可能解释的不是很清楚，不妨来举个例子说明一下我的想法。就比如说明天早上太阳在东边升起，西边落下。那么关于这个事件，它发生的可能性是一定的，当然也有可能今天晚上宇宙爆炸，这说不好，但是我们不考虑这种情况的发生。太阳东升西落这个就是明天一定要发生的没有其他的可能性而言，他的信息量就是0，因为它不包含其它可能发生的事情。再来看一个例子，比如，我明天去钓鱼。那么关于我去钓鱼这个事情，他可能发生，也可能不发生，当然不发生的概率是很大的，那么我不去钓鱼去干什么？可能去爬山，看电影，约会等等，这个有非常多的可能，由此可见在关于这一个事件的预测下，包含了很多的其它事情，所以他的信息量的非常大的，这个量就是我认为的，可能发生的事件数量。这里我们给出信息量的计算公式：

$$
Info=-log_2(p_k)
$$


### Entropy：

​				熵这个概念第一次接触的时候是在上高中化学课的时，他表示的是在反应过程中某个体系的混乱程度。在信息熵中，他也是表示的是一种混乱程度。其实这个，我个人也做过一些更深刻的思考，这个所谓的混乱，其实是相对于我们观察来说的，就像如果我们把一把红豆和一把绿豆放在一个碗里面，自然而然的红豆和绿豆他们会趋向于变得混在一起而不是分层。在我们看来红豆绿豆分层的话，这个是均匀的，但是如果我们以一整个自然体系来看，均匀不正是应该每一个地方有红豆和绿豆的概率是一样的嘛？这和我们认为的均匀是相悖的，但是没关系，只是我的一个小小的思考。我们这里的信息熵，表示的是信息的混乱程度。其实就是相对于某一个事件发生的所有可能性的对于每一种可能性的信息量的加权平均数，所谓信息量，我个人感觉他表示的就是一种混乱程度，但是他是相对于某一种可能性的，而相对于事件，就把每一种可能性的混乱程度加权平均，这很容易理解，这里我们也给出他的计算公式：

$$
Ent(D)=-ΣP_klog_2(P_k)
$$


### 		Attribute division：

​				

#### 							Entropy Gain(ID3):

​				在ID3决策树中，属性的划分方法采用的是信息增益的方法，信息增益很好理解。首先这个增益坑定是针对某一个事件的，不可能针对的是某一个事件中的某一种可能性。那么既然是针对于某一个事件的，坑定要用两个信息熵的差来计算这个增益，毫无疑问的，坑定是某一个事件之前有一个信息熵，然后经过处理之后，又有一个信息熵。这两个信息熵的差就是我们所谓的信息增益。那么对于我们的西瓜来说，这个事件有两种发生的可能性，要么是好瓜，要么是坏瓜。这个时候我们拿到一个数据集，我们可以很容易的计算出这批西瓜里面好瓜的概率和坏瓜的概率，争对西瓜是好还是坏，我们找到了他的每一种可能性发生的概率，根据前面的信息熵的公式，我们不难算出没有划分之前的西瓜好坏的信息熵，然后我们根据数据集中西瓜的每一种属性来划分数据集，划分后，我们依然可以得到每一个划分后的数据集的信息熵，然后我们把每一个数据集的信息熵加起来，当然这里的相加是加权的一个相加，加起来之后，我们就能得到划分后数据集的信息熵，我们把这两个信息熵相减，得到一个信息增益，就这样计算出按照每一个属性划分的信息增益，我们选取信息增益最大的作为我们的划分依据，因为，信息增益大了，混乱程度减小的就多了嘛，这很容易理解。之后我们在计算划分后每一个数据集的每一个属性的信息增益，得到最佳属性，直到最后没有属性可以划分，或者已经信息增益为0了。由此我们可以得出信息增益的计算公式：

$$
EntropyGain(X,x_i)=Entropy(X)-Σ\frac{x_i}{X}Entropy(x_i)
$$


#### 							Gain_Ratio(C4.5):

​				在C4.5决策树中，采用信息增益率来作为划分的依据。这个主要是为了解决ID3决策树对于可取值数目较多的属性有 所偏好,或者偏好于某个属性的每个取值的样本数 非常少的情况。就比如我们把它的编号x<sub>1</sub>作为一个划分的依据，那么

$$
Entropy(x_1)=Σ\frac{x_1}{X}Entropy(x_i)=n(\frac{1}{n}(-1log_21))=0
$$

很显然这个时候信息增益是最大的，因此为了解决这一冲突，我们在信息增益的基础上，让它除以一个相对的数字，来遏制这种结果，我们希望这个数字随着某一个事件可以发生可能性的变多而增大，这个数字就是IV，他的计算方式和信息熵的计算方式是差不多的，我们将某一个属性作为一个事件，这个属性的取值作为事件发生的可能性，计算这个属性的信息熵，就是IV了，来看它的计算公式：

$$
IV(A,a_i)=-Σ\frac{a_i}{A}log_2\frac{a_i}{A}
$$

这里的‘A’表示在该属性下所有的可能性的数量，a<sub>i</sub>表示属性a<sub>i</sub>的数量然后信息增益率就是：

$$
Gain_Ratio(X,x_i)=\frac{EntropyGain(X,x_i)}{IV(x_i,x_{ij})}
$$


#### 							Gini(C5.0):

​				在C5.0决策树中，划分属性采用基尼系数，这个其实就是按照信息熵来划分的，我们直接来看公式：

信息熵的公式：

$$
Ent(D)=-ΣP_klog_2(P_k)
$$

基尼系数的公式：

$$
Gini(D)=-ΣP_K*(1-P_k)=1-Σ(P_k)^2
$$

1- x是- log2 x的近似，实际上是后者的泰勒展开的一阶近似。 所以，基尼值实际上是信息熵的近似值。然后其它的和信息增益的步骤是一样的：

$$
GiniIndex(X,x_i)=-Σ\frac{x_i}{X}Gini(x_i)
$$


### 		Prune：

​			为了避免过拟合，我们需要对决策树进行剪枝处理，这里有预剪枝（Want To Prune）和后剪枝（Post Prune）两种方法

#### 							Want To Prune:

​			预剪枝，从名字就可以看出来，是先进行剪枝，在生成决策树，但是，我们连决策树都没有，怎么剪枝？那就是在决策树生成的过程中进行剪枝喽！假设现在我们要考虑一个结点该不该划分，那我们应该有一个度量的指标来确定这个结点该不该划分。这个指标用来对比划分前后决策树的好坏，那么判断一个决策树的好坏，自然而然的就想到‘精度’这个量：

$$
Acc(X,Y)=\frac{Σ(f(x_i)==y_i)}{N}
$$

既然要对比前后，那么坑定不能只在一个数据集上进行操作，所以这里就需要将数据集划分为训练集和测试集，在划分某一个结点的时候，考虑划分前后，决策树在测试集上的精度是否增加。

#### 							Post Prune:

​			在预剪枝生成的决策树中，可以降低过拟合风险，显著减少训练时间和测试时间开销，但是，有些分支的当前划分虽然不能提升 泛化性能，但在其基础上进行的后续划分却有可 能导致性能显著提高。预剪枝基于“贪心”本质 禁止这些分支展开，带来了欠拟合风险，由此为了避免预剪枝带来的欠拟合风险，我们可以采用后剪枝的方法，也就是先生成一颗决策树，再来对其结点进行评估，决定其是否去留。依然采用精度作为评估的依据，进行自底而上的剪枝操作。



### 		Data Processing:



#### 						Continuous Value Processing:

​			在构建决策树的时候我们拿到的数据集的属性一般情况下都是一些离散的值就像西瓜数据集中的根蒂，颜色等等，但是，不乏有一些数据集也会有着一些连续的值，比如西瓜的甜度，含糖率等等，这些连续的值，他们很少存在说几个值相同的情况，那么如果按照这一个属性划分的话，我们得到的结果将会很惨淡，很大几率会让我们的决策树过拟合，所以我们需要对这一个连续的值进行处理。这里我们了解到了二分法。

​			二分，顾名思义，是将这个连续值的属性分为两个类别，那么最直观的方法就是我们找到这些连续值中间的某一个值T，让属性中大于T的为A类，小于等于T的为另一类。假设我们现在有一个属性值为连续值的属性X，那么我们就需要找到这样的一个值T：

$$
f(X_i)=\begin{cases} 
		1, & X_i>T\\ 
		0, & X_i<=T
\end{cases}
$$


那么该如何找到这个值呢？其实很简单，我们将属性X中所有的属性值按照从小到大的顺序进行排练，然后用X<sub>i+1</sub>-X<sub>i</sub>,得到一个拥有Length(X)-1的集合，我们让这个集合为A，A中的元素为A<sub>i</sub>,我们让T的值一个一个的去等于A<sub>i</sub>,这样呢，我们就能得到Length（X）-1种划分的方法，来将这个连续值的属性划分为两类。下面我们要进行的就是要从这些类中找到一个最合适的划分方法，那么这个就和前面的属性划分方法差不多了，就是计算他的熵增，选取一个能使熵增最大的A<sub>i</sub>,让它等于T，这样就得到了我们期望得到的值。这里我们给出他的计算公式：

$$
T=max(EntropyGain(X,A,A_i))=max[Entropy(X)-Σ\frac{Σf(A_i)}{A}*Entropy(Σf(A_i))]
$$


#### 						Missing Value Handling:

#### End：

注：以下实验结果均以ID3决策树为例，并且没有进行剪枝处理。

#### 		Data：

![data](.\data.png)

#### Analysis:

![end](.\end.png)




$$
Label=['否', '否', '是', '是']
$$

$$
Prediect=['否', '是', '是', '是']
$$



<h4>分类结果混肴矩阵:</h4>   <!--标题-->
<table border="1" width="500px" cellspacing="10">
<tr>
  <th align="left">真实情况\预测结果</th>
  <th align="center">正例</th>
  <th align="right">反例</th>
</tr>
<tr>
  <td>正例</td>
  <td>TP=2</td>
  <td>FN=0</td>
</tr>
<tr>
  <td>反例</th>
  <td>FP=1</td>
  <td>TN=1</td>
</tr>
</table>



$$
Accuracy={(TP+TN)\over(TP+TN+FN+FP)}=0.75
$$

$$
Precision={TP\over(TP+FP)}=0.67
$$

$$
Recall={TP\over(TP+FN)}=1
$$




#### Code:

​			这次实验的理论虽然简单，但是实验的逻辑着实有些复杂，在这里我定义了一个交Decision Tree的类。这个类可以实现ID3，C4.5，以及CART决策树的构建以及加载使用。

##### Coding：

​			这个类的第一个函数，就是初始化函数，这里呢，有两个参数，__第一个参数就是数据集的路径。默认的路径是 './watermelon2.0.csv'，当然也可以自己传入路径__，数据集的格式就是上面数据集图表去除最后一列。然后__第二个参数是关于决策树的选择，可以在‘ID3'，’C4.5','CART‘三个参数中选择一个，分别代表构建和使用不同的决策树__。

```python
def __init__(self,DataSet_Path=r'./watermelon2.0.csv',Mode='ID3'):
    self.DataSet_Path=DataSet_Path
    self.Mode=Mode
```

​			

​			第二个函数就是处理数据集的一个函数。__我们对数据集进行正例和反例的分层采样__，构建训练集(TrainDataset）和测试集(TestDataSet),同时为了方便操作，我们也返回了他的整体数据集（DataSet）。数据集返回的格式我放在代码段的结尾。

​			具体是怎么实现的，其实就是不断的变换类型，切片就行了，没有什么难的地方，反正就是这个作用，具体的注释就不写了，因为代码太长了，写起注释会显得很混乱。

```python
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
####_TestDataSet=[['浅白', '蜷缩', '浊响', '模糊', '平坦', '硬滑', '否'], ['乌黑', '稍蜷', '沉闷', '稍糊', '稍凹', '硬滑', '否'], ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', '是'], ['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', '是']]
```

​			第三个函数，就是我们用测试集来__构建决策树的一个函数__了，这个函数的逻辑实在有点复杂，他能运行出来，一半考我对代码以及理论的理解，另一半考运气，不断的修改，尝试。好在写出来了，但是这里__返回的参数，不是决策树标准的输入与输出，而是一个列表__。不要问我为什么不返回那个字典，因为我不会！！！对字典的操作实在是四舍五入约等于零的。输出的格式我放在代码段的结尾。



```python
def ID3_Tree(self,_TrainDataset=None,_Parameter=None):  #属性划分方法采用Info_Gain方法

    if _TrainDataset==None:
        _TrainDataset=self._TrainDataset
    if _Parameter==None: 
        _Parameter=[]	#这个参数用来记录决策树的信息

   ##########################计算熵增#############################################
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
#######################################################################################
#######具体熵增的计算方式在上面的公式推导，这里的代码就是实现熵增的计算############################
######################################################################################

#####################这里利用计算出来的熵增的集合来对数据集进行划分，然后递归重复操作##########
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
###############由于细节比较难以言喻，并且是多次修改，所有在写的时候没有写注释#####################
    return _Parameter

###Parameter=['纹理', '模糊', '否', '纹理', '清晰', '根蒂', '硬挺', '否', '根蒂', '稍蜷', '色泽', '乌黑', '触感', '硬滑', '是', '触感', '软粘', '否', '色泽', '青绿', '是', '根蒂', '蜷缩', '是', '纹理', '稍糊', '色泽', '乌黑', '是', '色泽', '浅白', '否', '色泽', '青绿', '否']
```

​				__剩下的两个函数C4.5和CART，和上面的ID3只有划分属性的计算方法不同，其余的完全一样，这里就不搬出来分析了。__



​				然后就是Packing_Department函数，__这个函数是对构建的决策树的返回值进行处理和使用的。由于他返回的是一个单维度的列表，所以我们不能直接使用，我们需要对他进行一些处理。我的处理方法就是，把这颗决策树的所有从上到下的枝条都放在一个大集合中，每一个小枝条就是一个小集合。然后让测试的数据集来匹配这些集合，得到预测值__。具体的实现细节，我说不清楚，因为这一点我写了三天，整整三天！！！不断的改，不断的删，重写，太折磨了，好在写出来了，如果再写不出来，我就要奔溃了。



```python
def Packing_Department(self,_TestDataSet=None):
    if _TestDataSet==None:
        _TestDataSet=self._TestDataSet

    if self.Mode=='ID3':
        Attribute_Parameter=Decision_Tree.ID3_Tree(self)
        Attribute_Parameter=sum(Attribute_Parameter,[])
        StarCar,Star =[],[]
        print(Attribute_Parameter)
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
```

​							__这个就是关于决策树的核心代码的一些介绍了，虽然可能介绍了和没有介绍一样__。



##### All Code:

```python
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
#########################################################################################
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

#########################################################################################
#########################################################################################
#########################################################################################

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

###########################################################################################

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

###########################################################################################
###########################################################################################
###########################################################################################

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
##########################################################################################
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


#######################################################################################
#																					###	
#######################################################################################
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

##########################################################################################
    def Drawing(self):
        t.screensize(800, 600)
        t.pensize(2)  # 设置画笔的大小
        t.speed(10)  # 设置画笔速度为10
        t.write("这个图画起来比较繁琐，\n在网上找了一些教程，\n都是利用sklearn进行画图，\n当然也有以标准输入输出画图的，但是\n显然我的输出\n不是标准决策树输出，所以打开画图软件，画图\n", align="center", font=("楷体", 16, "bold"))
        t.end_fill()
        t.done()
###########################################################################################
    def Packing_Department(self,_TestDataSet=None):
        if _TestDataSet==None:
            _TestDataSet=self._TestDataSet
##########################################################################################
        if self.Mode=='ID3':
            Attribute_Parameter=Decision_Tree.ID3_Tree(self)
            Attribute_Parameter=sum(Attribute_Parameter,[])
            StarCar,Star =[],[]
            print(Attribute_Parameter)
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
###########################################################################################
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
############################################################################################
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

##########################################################################################
A=Decision_Tree(Mode='ID3')
A.Create_DataSet()
A.Packing_Department()
```
