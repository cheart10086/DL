import matplotlib.pyplot as plt
import numpy as np

x=np.linspace(-1,1,50)
y=2*x+1
help(plt.plot)
# plt.plot(x,y,'v')
# plt.show()
#平滑折线的示例
# plt.plot(xnew, ynew, marker='.', markevery=markevery, ls='-', label='yy', c='purple',
# linewidth=1.0, ms=6, mfc='purple', mec='purple', mew=3)

#设置横纵坐标的刻度范围
plt.xlim((0, 10200))   #x轴的刻度范围被设为a到b
plt.ylim((0, 0.32))    #y轴的刻度范围被设为a'到b'

#设置X轴和Y轴的标签名
plt.xlabel('Number of iterations', fontproperties=myfont, fontsize=18)   #(label, 字体,  字体大小)
plt.ylabel('Accuracy', fontproperties=myfont, fontsize=18)

#将X轴的刻度设为epoch对应的值（也就是只显示epoch对应的那几个刻度值）
del x[0]  #如果XY轴都是从0开始，可以只保留一个0，另一个删掉

#设置y轴的显示刻度线的几个值
#ytick = np.array([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30])    #形式1
ytick = np.arange(start=0, stop=35, step=5) / 100  #[start, stop)   形式2

#若是想用其他字符(串)代替这些点的值，可以用到它,。比如这个，我将Y轴的0.05、0.15、0.25用空字符串"代替
ylabel = [0, '', 0.1, '', 0.2, '', 0.3]
xlabel = []
plt.xticks(x,  fontsize=10)   #对于X轴，只显示x中各个数对应的刻度值
#plt.yticks(ytick, fontsize=10)  #若是不用ylabel，就用这项
plt.yticks(ytick, ylabel, fontsize=10)    #根据要求决定是否用这个， 显示的刻度值：ylabel的值代替ytick对应的值

# 不显示坐标轴的值
# plt.xticks(())
# plt.yticks(())

ax = plt.gca()  #返回坐标轴
#这些都是可选操作
# ax.spines['right'].set_color('none')   #将右边框的颜色设为无
# ax.spines['right'].set_visible(False)  #让左边框不可见
# ax.spines['top'].set_color('none')   #将上边框设为无色
# ax.spines['left'].set_color('green')   #将左边框的颜色设为绿色
# ax.xaxis.set_ticks_position('bottom') #将x轴的刻度值写在下边框的下面
# ax.yaxis.set_ticks_position('left')  #将y轴的刻度值写左边框的左面
# ax.spines['bottom'].set_position(('data', 0))   #将下边框的位置挪到y轴的0刻度处
# ax.spines['left'].set_position(('data', 50))  #将左边框挪到x轴的50刻度处

ax.tick_params(axis='x', tickdir='in', labelrotation=20)   #坐标轴的刻度线及刻度值的一些参数设置，详见文末
ax.tick_params(axis='y', tickdir='in')

#可选部分
axlinewidth = 1.2
axcolor = '#4F4F4F'
ax.spines['bottom'].set_linewidth(axlinewidth); #设置底部坐标轴的粗细
ax.spines['left'].set_linewidth(axlinewidth); #设置左边坐标轴的粗细
ax.spines['right'].set_linewidth(axlinewidth); #设置右边坐标轴的粗细
ax.spines['top'].set_linewidth(axlinewidth); #设置上部坐标轴的粗细
ax.spines['bottom'].set_color(axcolor)  #设置坐标轴颜色
ax.spines['left'].set_color(axcolor)
ax.spines['right'].set_color(axcolor)
ax.spines['top'].set_color(axcolor)



#显示图例
'''loc:图例位置， 
   fontsize：字体大小，
   frameon：是否显示图例边框，
   ncol：图例的列的数量，一般为1,
   title:为图例添加标题
   shadow:为图例边框添加阴影,
   markerfirst:True表示图例标签在句柄右侧，false反之，
   markerscale：图例标记为原图标记中的多少倍大小，
   numpoints:表示图例中的句柄上的标记点的个数，一半设为1,
   fancybox:是否将图例框的边角设为圆形
   framealpha:控制图例框的透明度
   borderpad: 图例框内边距
   labelspacing: 图例中条目之间的距离
   handlelength:图例句柄的长度
   bbox_to_anchor: (横向看右，纵向看下),如果要自定义图例位置或者将图例画在坐标外边，用它，比如bbox_to_anchor=(1.4,0.8)，这个一般配合着ax.get_position()，set_position([box.x0, box.y0, box.width*0.8 , box.height])使用
   用不到的参数可以直接去掉,有的参数没写进去，用得到的话加进去     , bbox_to_anchor=(1.11,0)
'''
# box = ax.get_position()
# ax.set_position([box.x0, box.y0, box.width*0.8 , box.height]) #若是将图例画在坐标外边，如果放在右边，一般要给width*0.8左右的值，在上边，要给height*0.8左右的值
ax.legend(loc='lower right', fontsize=12, frameon=True, fancybox=True, framealpha=0.2, borderpad=0.3,
           ncol=1, markerfirst=True, markerscale=1, numpoints=1, handlelength=3.5)    #可以用ax.legend,也可以用plt.legend

#保存图像
#plt.savefig('E:/1.png', dpi=100, bbox_inches='tight')  #, dpi=100

#显示图
plt.show()

'''
plt.legend(loc)中的loc取值：
 0:'best'
 1: 'upper right'
 2: 'upper left'
 3:'lower left'
 4: 'lower right'
 5: 'right'
 6: 'center left'
 7: 'center right'
 8: 'lower center'
 9: 'upper center'
 10: 'center'
'''


'''
Axes.tick_params(axis='both', **kwargs)
参数:
axis :  {'x', 'y', 'both'} 选择对哪个轴操作，默认是'both'
reset :  bool If True, set all parameters to defaults before processing other keyword arguments. Default is False.
which :  {'major', 'minor', 'both'} 选择对主or副坐标轴进行操作
direction/tickdir : {'in', 'out', 'inout'}刻度线的方向
size/length : float, 刻度线的长度
width :  float, 刻度线的宽度
color :  刻度线的颜色
pad :  float, 刻度线与刻度值之间的距离
labelsize :  float/str, 刻度值字体大小
labelcolor : 刻度值颜色
colors :  同时设置刻度线和刻度值的颜色
zorder : float Tick and label zorder.
bottom, top, left, right : bool, 分别表示上下左右四边，是否显示刻度线，True为显示
labelbottom, labeltop, labelleft, labelright :bool, 分别表示上下左右四边，是否显示刻度值，True为显示
labelrotation : 刻度值逆时针旋转给定的度数，如20
gridOn: bool ,是否添加网格线； grid_alpha:float网格线透明度 ； grid_color: 网格线颜色;  grid_linewidth:float网格线宽度； grid_linestyle: 网格线型 
tick1On, tick2On : bool分别表表示是否显示axis轴的(左/下、右/上)or(主、副)刻度线
label1On,label2On : bool分别表表示是否显示axis轴的(左/下、右/上)or(主、副)刻度值

ALL param:
['size', 'width', 'color', 'tickdir', 'pad', 'labelsize', 'labelcolor', 'zorder', 'gridOn', 'tick1On', 'tick2On', 
'label1On', 'label2On', 'length', 'direction', 'left', 'bottom', 'right', 'top', 'labelleft', 'labelbottom', 'labelright',
 'labeltop', 'labelrotation', 'grid_agg_filter', 'grid_alpha', 'grid_animated', 'grid_antialiased', 'grid_clip_box', 
 'grid_clip_on', 'grid_clip_path', 'grid_color', 'grid_contains', 'grid_dash_capstyle', 'grid_dash_joinstyle', 'grid_dashes', 
 'grid_drawstyle', 'grid_figure', 'grid_fillstyle', 'grid_gid', 'grid_label', 'grid_linestyle', 'grid_linewidth', 'grid_marker', 
 'grid_markeredgecolor', 'grid_markeredgewidth', 'grid_markerfacecolor', 'grid_markerfacecoloralt', 'grid_markersize', 
 'grid_markevery', 'grid_path_effects', 'grid_picker', 'grid_pickradius', 'grid_rasterized', 'grid_sketch_params', 'grid_snap', 
 'grid_solid_capstyle', 'grid_solid_joinstyle', 'grid_transform', 'grid_url', 'grid_visible', 'grid_xdata', 'grid_ydata', 
 'grid_zorder', 'grid_aa', 'grid_c', 'grid_ls', 'grid_lw', 'grid_mec', 'grid_mew', 'grid_mfc', 'grid_mfcalt', 'grid_ms']
'''
