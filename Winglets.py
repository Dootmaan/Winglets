import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from scipy.optimize import curve_fit

size=500

def distance(x1,y1,x2,y2):
    return np.sqrt((x1-x2)**2+(y1-y2)**2)


class PlotDemo:
    def __init__(self, m, n):
        self._net = None
        self.m = m
        self.n = n

    def _set_default_figure(self, m, n):
        # plt.figure(figsize=(8,8))
        ax = plt.subplot(111)  #注意:一般都在ax中设置,不再plot中设置
        #设置主刻度标签的位置,标签文本的格式
        # xmajorLocator = MultipleLocator(10)  #将x主刻度标签设置为20的倍数
        # xmajorFormatter = FormatStrFormatter('%1.1f')  #设置x轴标签文本的格式
        # xminorLocator = MultipleLocator(1)  #将x轴次刻度标签设置为5的倍数
        # ymajorLocator = MultipleLocator(10)  #将y轴主刻度标签设置为0.5的倍数
        # ymajorFormatter = FormatStrFormatter('%1.1f')  #设置y轴标签文本的格式
        # yminorLocator = MultipleLocator(1)  #将x轴次刻度标签设置为5的倍数
        ax.set_xlim(0, m)
        ax.set_ylim(0, n)
        # ax.xaxis.set_major_locator(xmajorLocator)
        # ax.xaxis.set_major_formatter(xmajorFormatter)
        # ax.yaxis.set_major_locator(ymajorLocator)
        # ax.yaxis.set_major_formatter(ymajorFormatter)
        #显示次刻度标签的位置,没有标签文本
        # ax.xaxis.set_minor_locator(xminorLocator)
        # ax.yaxis.set_minor_locator(yminorLocator)
        ax.xaxis.grid(True, which='minor')  #x坐标轴的网格使用主刻度
        ax.yaxis.grid(True, which='minor')  #y坐标轴的网格使用次刻度

    def show_source(self):
        self._set_default_figure(self.m, self.n)

        if self._net is None:
            raise Exception
        net = self._net.net_info
        shape = net.shape
        xlen = shape[0]
        ylen = shape[1]
        for i in range(xlen):
            for j in range(ylen):
                if net[i][j] > 0:
                    plt.plot(i, j, 'g+')
        plt.show()

    def draw_contour(self, _x, _y, _m, _n,style='r+'):
        drawing_list=[]
        self._set_default_figure(self.m, self.n)
        net = self._net.net_info
        # shape = net.shape
        # xlen = shape[0]
        # ylen = shape[1]
        # for i in range(xlen):
        #     for j in range(ylen):
        #         if net[i][j] > 0:
        #             plt.plot(i,j,'g+')

        net = self._net
        utils = MarchSquareUtlis(net)
        lines = utils.trancing_contours()

        height, width = net.net_info.shape
        arr = net.net_info
        idx = 0
        track_x=[]
        track_y=[]
        for i in range(height - 1):
            for j in range(width - 1):
                x, y = i, j
                count, v1, v2, v3, v4, v5, v6, v7, v8 = lines[idx]
                idx = idx + 1
                if count == 0 or x < _x - _m or x > _x + _m or y < _y - _n or y > _y + _n:
                    continue
                if count == 1:
                    x1 = x + v1
                    y1 = y + v2
                    x2 = x + v3
                    y2 = y + v4
                    # track_x.append(x1)
                    # track_x.append(x2)
                    # track_y.append(y1)
                    # track_y.append(y2)
                    track_x.append((x1+x2)/2)
                    track_y.append((y1+y2)/2)
                    # plt.plot([x1, x2], [y1, y2], style,linewidth=1, markersize=1)
                if count == 2:
                    x1 = x + v1
                    y1 = y + v2
                    x2 = x + v3
                    y2 = y + v4
                    # track_x.append(x1)
                    # track_x.append(x2)
                    # track_y.append(y1)
                    # track_y.append(y2)
                    track_x.append((x1+x2)/2)
                    track_y.append((y1+y2)/2)
                    # plt.plot([x1, x2], [y1, y2], style,linewidth=1, markersize=1)
                    x1 = x + v5
                    y1 = y + v6
                    x2 = x + v7
                    y2 = y + v8
                    # track_x.append(x1)
                    # track_x.append(x2)
                    # track_y.append(y1)
                    # track_y.append(y2)
                    track_x.append((x1+x2)/2)
                    track_y.append((y1+y2)/2)
                    # plt.plot([x1, x2], [y1, y2], style,linewidth=1, markersize=1)
        drawed=[(_x,_y)]
        if(len(track_x)>0):
            for i in range(len(track_x)):
                drawing_list.append((track_x[i],track_y[i]))

            nearest=drawing_list[0]
            dis=0
            for whatever in range(2):
                ox=_x
                oy=_y
                while(True):
                    min_dis=1000
                    for (x1,x2) in drawing_list:
                        if (x1,x2) in drawed:
                            continue

                        dis=distance(x1,x2,ox,oy)
                        if(dis<min_dis):
                            min_dis=dis
                            nearest=(x1,x2)
                        
                    if(min_dis>2):
                        break
                    else:
                        x1,y1=nearest
                        # plt.plot(np.linspace(ox,x1,10),np.linspace(oy,y1,10),style,linewidth=1, markersize=1)
                        plt.plot([ox,x1],[oy,y1],style,linewidth=1, markersize=1)
                        drawed.append((x1,y1))
                        ox=x1
                        oy=y1

    def set_net_info(self, net_info):
        self._net = net_info


class PlotInfo(object):
    def __init__(self, m, n):
        self.arr = self._gen_random(m, n)

    @property
    def net_info(self):
        return self.arr

    def _gen_random(self, m, n):
        return np.zeros((m, n), dtype='double')

    def set_arr(self, array, value):
        arr = np.zeros((len(array), len(array[0])))
        for i in range(len(array)):
            for j in range(len(array[0])):
                if array[i][j]>value :
                    arr[i][j] = 1
                else:
                    arr[i][j] = 0
        self.arr = arr



#clock-wise, top-left|top-right|bottom-right|bottom_left
def get_retangle_bit(v1, v2, v3, v4):
    return v1 << 3 | v2 << 2 | v3 << 1 | v4


#shift relative to top-left
def get_retangle_shift(bitval):
    if bitval == 0 or bitval == 15:
        return (0, None, None, None, None, None, None, None, None)
    if bitval == 1 or 14:
        return (1, 0, 0.5, 0.5, 1, None, None, None, None)
    if bitval == 2 or bitval == 13:
        return (1, 0.5, 1, 1, 0.5, None, None, None, None)
    if bitval == 3 or bitval == 12:
        return (1, 0, 0.5, 1, 0.5, None, None, None, None)
    if bitval == 4 or bitval == 11:
        return (1, 0, 0.5, 1, 0.5, None, None, None, None)
    if bitval == 5:
        return (2, 0, 0.5, 0.5, 0, 0.5, 1, 1, 0.5)
    if bitval == 6 or bitval == 9:
        return (1, 0.5, 0, 0.5, 1, None, None, None, None)
    if bitval == 7 or bitval == 8:
        return (1, 0, 0.5, 0.5, 0, None, None, None, None)
    if bitval == 10:
        return (2, 0, 0.5, 0.5, 1, 0.5, 0, 1, 0.5)


class MarchSquareUtlis(object):
    def __init__(self, net):
        self.net = net

    def trancing_contours(self):
        ret = []
        height, width = self.net.net_info.shape
        arr = self.net.net_info
        for j in range(height - 1):  #不能理解，但必须先j再i，我试了很久才发现问题在这里，damn
            for i in range(width - 1):
                v1 = int(arr[i][j])
                v2 = int(arr[i + 1][j])
                v3 = int(arr[i + 1][j + 1])
                v4 = int(arr[i][j + 1])
                bitv = get_retangle_bit(v1, v2, v3, v4)
                net_shift = get_retangle_shift(bitv)
                ret.append(net_shift)
        return ret


#生成数据，点越多越慢，因为mei yimeiyi
class0 = 22
class1 = 20
class2 = 18

class_num = 3

total_num = class0 + class1 + class2

dimension = 2  #数据特征维度

data0 = []
data1 = []
data2 = []

miu_0 = [150, 60]
sigma_0 = [90, 100]
miu_1 = [400, 360]
sigma_1 = [90, 70]
miu_2 = [360, 120]
sigma_2 = [70, 70]

#由于每一类数据都需要被单独控制，所以只好这样分开写
for i in range(class0):
    X = np.random.randn(1, dimension)
    X[0, 0] = sigma_0[0] * X[0, 0] + miu_0[0]
    X[0, 1] = sigma_0[1] * X[0, 1] + miu_0[1]
    data0.append(X.squeeze())

for i in range(class1):
    X = np.random.randn(1, dimension)
    X[0, 0] = sigma_1[0] * X[0, 0] + miu_1[0]
    X[0, 1] = sigma_1[1] * X[0, 1] + miu_1[1]
    data1.append(X.squeeze())

for i in range(class2):
    X = np.random.randn(1, dimension)
    X[0, 0] = sigma_2[0] * X[0, 0] + miu_2[0]
    X[0, 1] = sigma_2[1] * X[0, 1] + miu_2[1]
    data2.append(X.squeeze())


data0 = np.array(data0)
data1 = np.array(data1)
data2 = np.array(data2)


def normDis(x, y, miu, sigma, rou):  #只是二维的正态分布
    return (1.0 / (2.0 * np.pi * sigma[0] * sigma[1] * np.sqrt(1 - rou**2))
            ) * np.exp(
                (-2 / (1 - rou**2)) * (((x - miu[0])**2) / sigma[0]**2 + (
                    (y - miu[1])**2) / sigma[1]**2 - 2 * rou * (
                        (x - miu[0]) * (y - miu[1]) / (sigma[0] * sigma[1]))))

# def func(x, a, b , h, k):
#     y = a *  + b
#     return y

#A*x.^2 + B*x.*y + C*y.^2 + D*x + E*y + F
def solve_ellipse(A,B,C,D,E,F):
            
    Xc = (B*E-2*C*D)/(4*A*C-B**2)
    Yc = (B*D-2*A*E)/(4*A*C-B**2)
        
    FA1 = 2*(A*Xc**2+C*Yc**2+B*Xc*Yc-F)
    FA2 = np.sqrt((A-C)**2+B**2)
    
    MA = np.sqrt(FA1/(A+C+FA2)) #长轴
    SMA= np.sqrt(FA1/(A+C-FA2)) if A+C-FA2!=0 else 0#半长轴
    
    if B==0 and F*A<F*C:
        Theta = 0
    elif B==0 and F*A>=F*C:
        Theta = 90
    elif B!=0 and F*A<F*C:
        alpha = np.arctan((A-C)/B)*180/np.pi
        Theta = 0.5*(-90-alpha) if alpha<0 else 0.5*(90-alpha)
    else:
        alpha = np.arctan((A-C)/B)*180/np.pi
        Theta = 90+0.5*(-90-alpha) if alpha<0 else 90+0.5*(90-alpha)
            
    if MA<SMA:
        MA,SMA = SMA,MA
            
    return [Xc,Yc,MA,SMA,Theta]



x = np.linspace(0, size, size+1)
y = np.linspace(0, size, size+1)
X, Y = np.meshgrid(x, y)  # 获得网格坐标矩阵

# # 进行颜色填充
# plt.contourf(X,Y,normDis(X,Y,(0,0),(1,1),2),8)
# #进行等高线绘制
# c = plt.contour(X,Y,normDis(X,Y,miu_0,sigma_0,0),20,colors='red',alpha=0.3)
# # 线条标注的绘制
# plt.clabel(c,inline=True,fontsize=10)

# d=plt.contour(X,Y,normDis(X,Y,miu_1,sigma_1,0),20,colors='blue',alpha=0.3)
# plt.clabel(d,inline=True,fontsize=10)

# d=plt.contour(X,Y,normDis(X,Y,miu_2,sigma_2,0),20,colors='green',alpha=0.3)
# plt.clabel(d,inline=True,fontsize=10)

# d=plt.contour(X,Y,normDis(X,Y,miu_3,sigma_3,0),20,colors='blue',alpha=0.3)
# plt.clabel(d,inline=True,fontsize=10)

base = normDis(miu_0[0], miu_0[1], miu_0, sigma_0, 0)
for point in data0:
    target = normDis(point[0], point[1], miu_0, sigma_0, 0)

    playground = PlotDemo(size, size)
    netinfo = PlotInfo(size, size)

    netinfo.set_arr(normDis(X, Y, miu_0, sigma_0, 0), target)  #这样会画出来一条等高线
    playground.set_net_info(netinfo)
    playground.draw_contour(point[0], point[1],
                            max([15, int(50 * (target / base))]),
                            max([15, int(50 * (target / base))]),'r+-')


base = normDis(miu_1[0], miu_1[1], miu_1, sigma_1, 0)
for point in data1:
    target = normDis(point[0], point[1], miu_1, sigma_1, 0)

    playground = PlotDemo(size, size)
    netinfo = PlotInfo(size, size)

    netinfo.set_arr(normDis(X, Y, miu_1, sigma_1, 0), target)  #这样会画出来一条等高线
    playground.set_net_info(netinfo)
    playground.draw_contour(point[0], point[1],
                            max([15, int(50 * (target / base))]),
                            max([15, int(50 * (target / base))]),'b+-')


base = normDis(miu_2[0], miu_2[1], miu_2, sigma_2, 0)
for point in data2:
    target = normDis(point[0], point[1], miu_2, sigma_2, 0)

    playground = PlotDemo(size, size)
    netinfo = PlotInfo(size, size)

    netinfo.set_arr(normDis(X, Y, miu_2, sigma_2, 0), target) #这样会画出来一条等高线
    playground.set_net_info(netinfo)
    playground.draw_contour(point[0], point[1],
                            max([15, int(50 * (target / base))]),
                            max([15, int(50 * (target / base))]),'g+-')


plt.scatter(data0[:, 0], data0[:, 1], c='red', s=30)
plt.scatter(data1[:,0],data1[:,1],c='blue',s=30)
plt.scatter(data2[:,0],data2[:,1],c='green',s=30)

plt.show()