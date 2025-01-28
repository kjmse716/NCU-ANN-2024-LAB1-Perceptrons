import numpy as np
import pandas
import random
import tkinter as tk
import tkinter.ttk
import matplotlib.pyplot as plt
import time
import os
from tkinter import filedialog
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.patches import Patch

#初始化全域變數
root = tk.Tk()


LR= tk.DoubleVar(value=0.1)
EP= tk.IntVar(value=10)
AC = tk.DoubleVar(value=100)
FP = tk.StringVar(value="E:\\Lab\\ANN\\HW1\\NN_HW1_DataSet\\basic\\2Clo.txt")
W = tk.StringVar(value=f"w:\n\n\n")
TrainingAC = tk.StringVar(value="epoch:0\nAC/epoch :0%")
TestingAC = tk.StringVar(value="\nTesting AC :0%")
learningRate= LR.get()
epoch= EP.get()
Acuracy = AC.get()

# 圖表初始化
fig1 = plt.figure()            
ax = fig1.add_subplot(111)  
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)

#選擇檔案按鈕
def select():
    global data,train_data,test_data,data_len,cata
    TrainingAC.set("epoch:0\nAC/epoch :0%")
    TestingAC.set("\nTesting AC :0%")
    FP.set(filedialog.askopenfilename()) 
    data = np.loadtxt(FP.get())
    data_len = len(data)
    cata = data[:,2]
    cata = np.unique(cata)
    cata = np.sort(cata)
    print(cata)

    split_data()
    data_update()



#切分訓練/測試集
def split_data():
    global train_data,test_data
    np.random.seed()
    train_data_index= random.sample(range(data_len),int(data_len/3*2))
    #print(train_data_index)
    train_data = data[train_data_index,:]
    test_data = np.delete(data,train_data_index,axis=0)


#更新圖表dots
def data_update():
    global line,line2,dots
    x=train_data[:,0]
    y=train_data[:,1]
    c = train_data[:,2]
    #print(test_data)
    ax.cla()
    dots=ax.scatter(x,y,c=c,cmap='viridis', vmin=cata[0], vmax=cata[1])
    line, = ax.plot([0,0], [0,0], color='red', label='線')
    
    # 創建對應 class 的 Patch，用來顯示 legend
    class_0_patch = Patch(color=dots.cmap(dots.norm(int(cata[0]))), label=f'Class {int(cata[0])}')
    class_1_patch = Patch(color=dots.cmap(dots.norm(int(cata[1]))), label=f'Class {int(cata[1])}')
    # 添加 legend，並將 class 0 和 class 1 對應顏色添加到圖例中
    ax.legend(handles=[class_0_patch, class_1_patch])
    

    canvas.draw()
    canvas.flush_events()
    time.sleep(0.1)    

    x=test_data[:,0]
    y=test_data[:,1]
    c = test_data[:,2]
    #print(test_data)
    ax2.cla()
    ax2.scatter(x,y,cmap='viridis')
    line2, = ax2.plot([0,0], [0,0], color='red', label='線')
    ax2.legend(handles=[class_0_patch, class_1_patch])
    canvas2.draw()
    canvas2.flush_events()
    time.sleep(0.1)    

#更新圖表預測線段
def w_update(w,line,canvas):
    c,a,b = w
    W.set(f"w:\n{float(c):.4f}\n{a:.4}\n{b:.4}\n")

    c=c*-1
    # 設定 x 的範圍，例如從 -10 到 10
    x_vals = np.linspace(-10, 10, 2)
    # 計算 y 的對應值
    y_vals = - (a / b) * x_vals - (c / b)
    line.set_data(x_vals,y_vals)
    #plt.plot(x_vals, y_vals, label=f'{a}x + {b}y + {c} = 0')
    canvas.draw()
    canvas.flush_events()
    time.sleep(0.1)
    return line

#訓練按鈕
def training():
    global w
    learningRate =LR.get()
    epoch= EP.get()
    Acuracy = AC.get()
    data_len = len(data)
    TrainingAC.set("epoch:0\nAC/epoch :0%")
    TestingAC.set("\nTesting AC :0%")


    np.random.seed()
    w = np.random.rand(3) * 0.01
    #w = np.insert(w,0,-1)
    print(f"randomw = {w}")
    w_update(w,line,canvas)


    print(w)
    print(train_data)
    train_accuracy = []
    for i in range(epoch):

        print(f"epoch = {i+1}")
        correct = len(train_data)
        for t_data in train_data:
            t_data = np.insert(t_data,0,-1)
            v = np.dot(w,t_data[0:3])
            print(f"v = {v}")
            y = cata[1] if v>0 else cata[0] if v<0 else 0
            #print(v)
            print(t_data)
            print(f"prediction = {y} , Answer = {t_data[3]} , w = {w}")
            if y !=t_data[3]:
                correct -= 1
                print("修改w")
                rate = learningRate if y < t_data[3] else -learningRate
                w = w + rate*t_data[0:3]
                print(f"rate = {rate}, {learningRate*t_data[0:3]}")
                print(f"w+ {rate*t_data[0:3]} , new w = {w}")
                w_update(w,line,canvas)
        train_accuracy.append(correct/len(train_data)*100)
        TrainingAC.set(f"epoch:{i+1}\nAC/epoch :{train_accuracy[-1]:.4}%")
        print(f"epoch: {i+1} 訓練正確率Accuracy{train_accuracy[-1]}%")
        
        if train_accuracy[-1] >=Acuracy:
            break
    print(train_accuracy)


#測試按鈕
def testing():
    global dots2,dots3
    test_result = []
    for t_data in test_data:
            t_data = np.insert(t_data,0,-1)
            v = np.dot(w,t_data[0:3])
            print(f"v = {v}")
            y = cata[1] if v>0 else cata[0] if v<0 else 0
            test_result.append(y)
            #print(v)
            print(t_data)
            print(f"prediction = {y} , Answer = {t_data[3]} , w = {w}")
    x=test_data[:,0]
    y=test_data[:,1]
    correct_index = []
    for v,k in enumerate(test_result):
        if test_data[v,2]==k:
            correct_index.append(v)
    TestingAC.set(f"\nTesting AC :{len(correct_index)/len(test_data)*100:.2f}%")
    test_result = np.array(test_result)
    ax2.cla()
    class_0_patch = Patch(color=dots.cmap(dots.norm(int(cata[0]))), label=f'Class {int(cata[0])}')
    class_1_patch = Patch(color=dots.cmap(dots.norm(int(cata[1]))), label=f'Class {int(cata[1])}')
    ax2.legend(handles=[class_0_patch, class_1_patch])

    dots2 = ax2.scatter(x[correct_index],y[correct_index],c=test_result[correct_index],marker="o",cmap='viridis', vmin=cata[0], vmax=cata[1])
    dots3 = ax2.scatter(np.delete(x,correct_index,axis=0),np.delete(y,correct_index,axis=0),c=np.delete(test_result,correct_index,axis=0),marker="x",cmap='viridis', vmin=cata[0], vmax=cata[1])
    line2,=ax2.plot([0,0], [0,0], color='red', label='線')

    w_update(w,line2,canvas2) 



#tkinter元件初始化
notebook = tk.ttk.Notebook(root)
frame1 = tk.Frame()
frame2 = tk.Frame()
notebook.add(frame1,text = "基本題")
notebook.add(frame2,text = "進階題")
#notebook.pack(fill = tk.BOTH,expand = True)
notebook.grid(row = 0,column=0,columnspan = 2,sticky=tk.W+tk.E+tk.S+tk.N)
Control_area = tk.Frame(root)
Control_area.grid(row = 0,column=2,sticky=tk.W+tk.E+tk.S+tk.N)
#Control_area.grid(row = 0,column=2,sticky=tk.W+tk.E+tk.S+tk.N)
training_area = tk.LabelFrame(frame1,text= "Training Set")
training_area.grid(row = 0,column=0)
testing_area = tk.LabelFrame(frame1,text= "Testing Set")
testing_area.grid(row = 0,column=1)


training_result_area = tk.LabelFrame(Control_area,text= "training_result")
training_result_area.grid(row = 0,column=0,sticky=tk.W+tk.E+tk.S+tk.N)
testing_result_area = tk.LabelFrame(Control_area,text= "testing_result")
testing_result_area.grid(row = 0,column=1,sticky=tk.W+tk.E+tk.S+tk.N)
variable_area = tk.LabelFrame(Control_area,text= "Variable")
variable_area.grid(row = 1,column=0,columnspan=2,sticky=tk.W+tk.E+tk.S+tk.N)
basic_action_area = tk.LabelFrame(Control_area,text="basic actions")
advance_action_area = tk.LabelFrame(Control_area,text="advance actions")


canvas = FigureCanvasTkAgg(fig1, master=training_area)  # A tk.DrawingArea.
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
canvas2 = FigureCanvasTkAgg(fig2, master=testing_area)  # A tk.DrawingArea.
canvas2.draw()
canvas2.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)


tk.Label(variable_area, text="learningRate").pack()
tk.Entry(variable_area, textvariable=LR).pack()
tk.Label(variable_area, text="max epoch").pack()
tk.Entry(variable_area, textvariable=EP).pack()
tk.Label(variable_area, text="stop accuracy").pack()
tk.Entry(variable_area, textvariable=AC).pack()
tk.Label(variable_area, textvariable=FP).pack()
#basic_action_area.pack(side='left',fill=tk.BOTH, expand=True)
#advance_action_area.pack(side='right',fill=tk.BOTH, expand=True)
basic_action_area.grid(row = 2,column=0,sticky=tk.W+tk.E+tk.S+tk.N)
advance_action_area.grid(row = 2,column=1,sticky=tk.W+tk.E+tk.S+tk.N)
tk.Button(basic_action_area,text="選擇檔案",command=select).pack()
tk.Label(training_result_area,textvariable=W).pack()
tk.Label(training_result_area,textvariable=TrainingAC).pack()
tk.Label(testing_result_area,textvariable=TestingAC).pack()

btn = tk.Button(basic_action_area, text='訓練/train',command=training)     # 建立 Button 按鈕
btn2 = tk.Button(basic_action_area, text='測試/test',command=testing)     # 建立 Button 按鈕
btn.pack()
btn2.pack()


#進階題實作
# 圖表初始化
fig3 = plt.figure()            
ax3 = fig3.add_subplot(111,projection = '3d')  
fig4 = plt.figure()            
ax4 = fig4.add_subplot(111,projection = '3d')  

#選擇檔案按鈕
def select_multi():
    global data,train_data,test_data,data_len,cata,n,dim
    TrainingAC.set("epoch:0\nAC/epoch :0%")
    TestingAC.set("\nTesting AC :0%")
    FP.set(filedialog.askopenfilename()) 
    data = np.loadtxt(FP.get())
    data_len = len(data)
    cata = data[:,-1]
    cata = np.unique(cata)
    cata = np.sort(cata)
    print(data.shape)
    split_data()
    n=int(cata[-1])
    if cata[0]==0 and n!=1:
        n+=1
    print(f"n={n}")
    dim = data.shape[1]-1
    if data.shape[1]<5:
        ndGraphInitialize()

#初始化三維圖表
def ndGraphInitialize():
    
    if data.shape[1]==4:
        print("顯示3維資料")
        ax3.cla()
        dots=ax3.scatter(train_data[:,0],train_data[:,1],train_data[:,2],c = train_data[:,3],vmin=-1,vmax=cata[-1])
        ax3.set_xlim(-20, 20)
        ax3.set_ylim(-20, 20)
        ax3.set_zlim(-20, 20)
        ax4.cla()
        ax4.scatter(test_data[:,0],test_data[:,1],test_data[:,2])
        ax4.set_xlim(-20, 20)
        ax4.set_ylim(-20, 20)
        ax4.set_zlim(-20, 20)
    elif data.shape[1]==3:
        ax3.cla()
        dots=ax3.scatter(train_data[:,0],train_data[:,1],c = train_data[:,2],vmin=-1,vmax=cata[-1])
        ax4.cla()
        ax4.scatter(test_data[:,0],test_data[:,1])

    # 創建對應 class 的 Patch，用來顯示 legend
    patchs = []
    '''
    patchs.append(Patch(color=dots.cmap(dots.norm(int(cata[-1]))), label=f'Class undefine prediction)'))
    if cata[0]==0:
        patchs.append(Patch(color=dots.cmap(dots.norm(int(cata[0]))), label=f'Class {int(cata[0])}'))
    '''
    for i in cata:
        patchs.append(Patch(color=dots.cmap(dots.norm(int(i))), label=f'Class {int(i)}'))
    # 添加 legend，並將 class 0 和 class 1 對應顏色添加到圖例中
    ax3.legend(handles=patchs)
    ax4.legend(handles=patchs)
    canvas3.draw()
    canvas3.flush_events()
    canvas4.draw()
    canvas4.flush_events()
    time.sleep(0.1)

#多維/群資料訓練
def ndTrain():
    global w,n,dim
    learningRate =LR.get()
    epoch= EP.get()
    Acuracy = AC.get()
    data_len = len(data)
    TrainingAC.set("epoch:0\nAC/epoch :0%")
    TestingAC.set("\nTesting AC :0%")


    np.random.seed()
    w =[]
    for k in range(n):
        w.append(np.random.rand(dim+1) * 0.01)
    #w = np.insert(w,0,-1)
    print(f"randomw = {w}")
    #w_update(w,line,canvas)
    if data.shape[1]<5:
        ndGraphInitialize()
    print(w)
    print(train_data)
    train_accuracy = []
    surface = []
    for i in range(epoch):

        print(f"epoch = {i+1}")
        correct = len(train_data)
        for t_data in train_data:
            t_data = np.insert(t_data,0,-1)
            code = []
            for k in range(n):
                code.append(np.dot(w[k],t_data[0:-1]))
                print(f"v{k} = {code[-1]}")
            code=list(map(lambda v:1 if v>0 else 0 if v<0 else 0,code))
            #print(v)
            print(t_data)
            Ans = [0]*n
            if cata[0] == 0:
                if n==1:
                    Ans[0]=t_data[-1]
                else:
                    Ans[int(t_data[-1])]=1
            else:
                Ans[int(t_data[-1])-1]=1
            print(f"prediction = {code} , Answer = {Ans} , w = {w}")
            if code !=Ans:
                correct -= 1
                print("修改w")
                for k in range(n):
                    if code[k]!=Ans[k]:
                        rate = learningRate if code[k] < Ans[k] else -learningRate
                        w[k] = w[k] + rate*t_data[0:-1]
                        print(f"rate = {rate}, {learningRate*t_data[0:-1]}")
                        print(f"w{k}+ {rate*t_data[0:-1]} , new w{k} = {w[k]}")
                        if dim ==3:
                            if len(surface)>k:
                                surface[k].remove()
                                surface[k] = nd_surface(w[k],ax3)
                            else:
                                surface.append(nd_surface(w[k],ax3))
                            canvas3.draw()
                            canvas3.flush_events()
                            time.sleep(0.1)
                        elif dim==2:
                            while len(surface)<=k:
                                surface.append(nw_update(w[k],ax3,-1,canvas3))
                            if len(surface)>k:
                                nw_update(w[k],ax3,surface[k],canvas3)
                #w_update(w,line,canvas)
        train_accuracy.append(correct/len(train_data)*100)
        TrainingAC.set(f"epoch:{i+1}\nAC/epoch :{train_accuracy[-1]:.4}%")
        print(f"epoch: {i+1} 訓練正確率Accuracy{train_accuracy[-1]}%")
        
        if train_accuracy[-1] >=Acuracy:
            break
    print(train_accuracy)

#三維更新圖表預測線段
def nw_update(w,ax,line,canvas):
    print(w)
    c,a,b = w
    W.set(f"w:\n{float(c):.4f}\n{a:.4}\n{b:.4}\n")

    c=c*-1
    # 設定 x 的範圍，例如從 -10 到 10
    x_vals = np.linspace(-10, 10, 2)
    x_vals = x_vals
    # 計算 y 的對應值
    y_vals = - (a / b) * x_vals - (c / b)
    if line == -1:
        newLine, = ax.plot([0,0],[0,0])
        return newLine
    else:
        line.set_data_3d(x_vals,y_vals,[0,0])
    #plt.plot(x_vals, y_vals, label=f'{a}x + {b}y + {c} = 0')
    canvas.draw()
    canvas.flush_events()
    time.sleep(0.1)

#三維更新圖表預測面 
def nd_surface(w,ax):
    d ,a, b, c = w  # 法向量
    # 定義 x 和 y 的範圍
    x = np.linspace(-10, 10, 2)
    y = np.linspace(-10, 10, 2)
    x, y = np.meshgrid(x, y)

    # 計算對應的 z 值，基於平面方程式 ax + by + cz = d
    z = (d - a * x - b * y) / c

    # 繪製平面
    return ax.plot_surface(x, y, z, alpha=0.5, rstride=100, cstride=100)

#多維/群資料測試
def ndTest():
    global w,n,dim
    learningRate =LR.get()
    epoch= EP.get()
    Acuracy = AC.get()
    data_len = len(data)
    TestingAC.set("\nTesting AC :0%")

    np.random.seed()
    print(f"w = {w}")
    correct_index = []
    predictions=[]
    for i,t_data in enumerate(test_data):
        t_data = np.insert(t_data,0,-1)
        code = []
        for k in range(n):
            code.append(np.dot(w[k],t_data[0:-1]))
            #print(f"v{k} = {code[-1]}")
        code=list(map(lambda v:1 if v>0 else 0 if v<0 else 0,code))
        sum = 0
        prediction = -1
        for k in range(n):
            sum+=code[k]
            if code[k]==1:
                prediction=k if cata[0] == 0 else k+1
        if sum>1:
            prediction=-1
        if n==1:
            prediction=code[0]
        predictions.append(prediction)
        print(f"code={code},prediction={prediction},Ans ={t_data[-1]}")
        if prediction == t_data[-1]:
            correct_index.append(i)
    predictions = np.array(predictions)
    TestingAC.set(f"\nTesting AC :{len(correct_index)/len(test_data)*100:.2f}%")
    if dim == 3:
        ax4.cla()
        ax4.scatter(test_data[correct_index,0],test_data[correct_index,1],test_data[correct_index,2],c=predictions[correct_index],marker='o',vmin=-1,vmax=cata[-1])
        incorrect_predictions = np.delete(predictions,correct_index,axis=0)
        x = test_data[:,0]
        y = test_data[:,1]
        z = test_data[:,2]
        ax4.scatter(np.delete(x,correct_index,axis=0),np.delete(y,correct_index,axis=0),np.delete(z,correct_index,axis=0),c=incorrect_predictions,marker='x',vmin=-1,vmax=cata[-1])
        ax4.set_xlim(-20, 20)
        ax4.set_ylim(-20, 20)
        ax4.set_zlim(-20, 20)
        canvas4.draw()
        canvas4.flush_events()
        time.sleep(0.1)
    elif dim==2:
        ax4.cla()
        ax4.scatter(test_data[correct_index,0],test_data[correct_index,1],c=predictions[correct_index],marker='o',vmin=-1,vmax=cata[-1])
        incorrect_predictions = np.delete(predictions,correct_index,axis=0)
        x = test_data[:,0]
        y = test_data[:,1]
        ax4.scatter(np.delete(x,correct_index,axis=0),np.delete(y,correct_index,axis=0),c=incorrect_predictions,marker='x',vmin=-1,vmax=cata[-1])
        ax4.set_xlim(-2, 6)
        ax4.set_ylim(-2, 6)
        ax4.set_zlim(-2, 6)
        surface = []
        while len(surface)<n:
            surface.append(nw_update(w[0],ax4,-1,canvas3))
        for k in range(n):
            nw_update(w[k],ax3,surface[k],canvas3)
        canvas4.draw()
        canvas4.flush_events()

        time.sleep(0.1)


tk.Button(advance_action_area,text="選擇多維檔案",command=select_multi).pack()
tk.Button(advance_action_area,text="多維/多群訓練",command=ndTrain).pack()
tk.Button(advance_action_area,text="多維/多群測試",command=ndTest).pack()
canvas3 = FigureCanvasTkAgg(fig3, master=frame2)  # A tk.DrawingArea.
canvas3.draw()
canvas3.get_tk_widget().grid(row = 0,column=0,sticky=tk.W+tk.E+tk.S+tk.N)
#canvas3.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
canvas4 = FigureCanvasTkAgg(fig4, master=frame2)  # A tk.DrawingArea.
canvas4.draw()
canvas4.get_tk_widget().grid(row = 0,column=1,sticky=tk.W+tk.E+tk.S+tk.N)

tk.mainloop()