import matplotlib.pyplot as plt
import numpy as np

#直线图
x = np.linspace(-3,3,50)
y1 = 2*x+1
y2 = x**2
#plt.figure是新建一个图
plt.figure()
plt.plot(x,y1)

#num=3是figure3的意思
plt.figure(num=3, figsize=(8,5))
plt.plot(x,y2)
plt.plot(x,y1,color='red',linewidth=1.0,linestyle='--')

#xlim,ylim相当于框选部分图
plt.xlim((-1,2))
plt.ylim((-2,3))
plt.xlabel('I am x')
plt.ylabel('I am y')

#替换x，y轴的标签.加$符号可以改变标签的字体
new_ticks=np.linspace(-1,2,5)
plt.xticks(new_ticks)
plt.yticks([-2,-1.8,-1,1.22,3],['$really\ bad$','$bad$','$normal$','$good$','$really\ good$'])

#设置坐标轴
#gca='get current position'
ax=plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.spines['bottom'].set_position(('data',0))
ax.spines['left'].set_position(('data',0))

#图例
plt.figure()
plt.plot(x,y2,label='up')
plt.plot(x,y1,color='red',linewidth=1.0,linestyle='--',label='down')
plt.legend()


#散点图
plt.figure()
n=1024
w=np.random.normal(0,1,n)
h=np.random.normal(0,1,n)
T=np.arctan2(w,h)#for color value
plt.scatter(w,h,s=75,c=T,alpha=0.5)
plt.xlim((-1.5,1.5))
plt.ylim((-1.5,1.5))
plt.xticks(()) ##不显示x轴标签
plt.yticks(())
plt.show()

#柱状图
plt.figure()
n=12
X=np.arange(12)
Y1=(1-X/float(n)*np.random.uniform(0.5,1.0,n))
Y2=(1-X/float(n)*np.random.uniform(0.5,1.0,n))

plt.bar(X,+Y1,facecolor='#9999ff',edgecolor='white')
plt.bar(X,-Y2,facecolor='#ff9999',edgecolor='white')

#x,y+0.05表示x轴位置不变，y轴位置向上移动0.05
#ha:horizontal alignmen 横向对齐方式，va:纵向对齐方式
for x,y in zip(X,Y1):
    print(x,y)
    plt.text(x,y+0.05,'%.2f'%y,ha='center',va='bottom') #%.2f表示保留2位小数。%y,代表输出的是y的值

for x,y in zip(X,Y2):
    print(x,y)
    plt.text(x,-y-0.05,'-%.2f'%y,ha='center',va='bottom')
    
plt.xlim(-1,n)
plt.ylim(-1.25,1.25)
plt.xticks(()) ##不显示x轴标签
plt.yticks(())
plt.show()

#3D图
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)
X=np.arange(-4,4,0.25)
Y=np.arange(-4,4,0.25)
X,Y=np.meshgrid(X,Y)
R=np.sqrt(X**2+Y**2)
#height value
Z=np.sin(R)
ax.plot_surface(X,Y,Z,rstride=1,cstride=1,cmap=plt.get_cmap('rainbow'))

ax.contourf(X,Y,Z,zdir='z',offset=-2,cmap='rainbow')#zdir='z'表示等高线从z轴压缩
ax.set_zlim(-2,2)#表示等高线的高度


#自定义布局-多合一显示
plt.figure()
plt.subplot(2,1,1) #2行，第一行只有一列
plt.plot([0,1],[0,1])

plt.subplot(2,3,4)#2行，第二行有3列，从上数第四个图片
plt.plot([0,1],[0,2])

plt.subplot(2,3,5)
plt.plot([0,1],[0,3])

plt.subplot(2,3,6)
plt.plot([0,1],[0,4])

#自定义布局-分格
import matplotlib.gridspec as gridspec
plt.figure()
ax1=plt.subplot2grid((3,3),(0,0),colspan=3,rowspan=1)
ax1.plot([1,2],[2,1])
ax1.set_title('ax1_title')
ax2=plt.subplot2grid((3,3),(1,0),colspan=2,rowspan=1)
ax3=plt.subplot2grid((3,3),(1,2),colspan=1,rowspan=2)
ax2=plt.subplot2grid((3,3),(2,0),colspan=1,rowspan=1)
ax2=plt.subplot2grid((3,3),(2,1),colspan=1,rowspan=1)




