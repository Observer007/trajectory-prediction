import sys
import importlib
importlib.reload(sys)

import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
mpl.rc('font', family='Times New Roman')

# ================================== data preparing ============================================================
BAR_WIDTH = 0.2
SPACE = 0.8
LEFT_PAD = 0.2
RIGHT_PAD = LEFT_PAD
INN_PAD = 0.05

y = [[132.499], [54.516], [43.974], [38.675], [28.138]]
y = np.array(y)

NUM_BAR, NUM_GROUP = y.shape
X_MAX = LEFT_PAD + RIGHT_PAD + SPACE*(NUM_GROUP-1)+BAR_WIDTH*NUM_GROUP*NUM_BAR + (NUM_BAR-1)*INN_PAD*NUM_GROUP


# ================================== settings =================================================================
figureName = 'barChart.pdf'

#-------------------------------------- style -----------------------------------------------------------------
# style option
#['bmh', 'seaborn-poster', 'seaborn-talk', 'seaborn-dark-palette', 'seaborn-muted', 'seaborn-colorblind',
# 'classic', 'seaborn-white', 'seaborn-pastel', 'dark_background', 'seaborn-ticks', 'seaborn-whitegrid',
# 'seaborn-darkgrid', 'fivethirtyeight', 'seaborn-notebook', 'seaborn-dark', 'grayscale', 'ggplot',
# 'seaborn-deep', 'seaborn-paper', 'seaborn-bright']

#print(plt.style.available)        # print all options
plt.style.use('seaborn-paper')   # set your style

#----------------------------------- font size ----------------------------------------------------------------
FONT_SIZE_LEGEND = 10
FONT_SIZE_AXIS = 18
FONT_SIZE_LABEL = 26
FONT_SIZE_TITLE = 26

#----------------------------------- axies setting ----------------------------------------------------------------
NUM_LINES = NUM_BAR
haf_group_len = (NUM_BAR*BAR_WIDTH + INN_PAD*(NUM_BAR-1))/2.0
X_LABEL_POSITION = [LEFT_PAD+i*SPACE+(2*i+1)*haf_group_len for i in range(NUM_GROUP)]

X_LIM = [0, X_MAX]
Y_LIM = [0,140]

XSTICK_LABEL= []

LINE_WIDTH_AXIS = 1.2

TITLE = ''
#LABEL_X = '$x$'
LABEL_X = None
LABEL_Y = 'rmse'
LABEL_TITLE = ''
FLAG_GRID = True

#----------------------------------- line setting ----------------------------------------------------------------
SYMBOL_EDGE_WIDTH = 1
BAR_EDGE_WIDTH = 1.2

# default color is black
LINE_COLOR = []
for i in range(NUM_LINES):
    LINE_COLOR.append(np.array([0, 0, 0])/255.0)

# LINE_COLOR[0] = np.array([86, 24, 27])/255
# LINE_COLOR[1] = np.array([214, 86, 42])/255
# LINE_COLOR[2] = np.array([60, 60, 60])/255
# LINE_COLOR[3] = np.array([90, 90, 90])/255

LINE_COLOR[0] = np.array([210, 210, 0])/255
LINE_COLOR[1] = np.array([102, 0, 0])/255
LINE_COLOR[2] = np.array([202, 98, 4])/255
LINE_COLOR[3] = np.array([232, 191, 191])/255
LINE_COLOR[4] = np.array([1, 46, 137])/255
# LINE_COLOR[5] = np.array([103, 12, 203])/255
# LINE_COLOR[6] = np.array([213, 71, 115])/255

# Hatch symbols
#[ '/' ,'\\' , '|', '-' ,'+' , 'x', 'o', 'O','.', '*']
HATCH_STYLE = []
for i in range(NUM_LINES):
    HATCH_STYLE.append('/')
HATCH_STYLE = ['..', '---', '///', '++', '\\\\\\']  # more symbols, the denser


#----------------------------------- legend setting ----------------------------------------------------------------
# legend position
# 'best' : 0, (only implemented for axes legends)
# 'upper right' : 1,
# 'upper left' : 2,
# 'lower left' : 3,
# 'lower right' : 4,
# 'right' : 5,
# 'center left' : 6,
# 'center right' : 7,
# 'lower center' : 8,
# 'upper center' : 9,
# 'center' : 10,
LEGEND_POSITION = 1
#LEGEND_POSITION = [-0.1,-0.1]

LEGEND_BOX_FLAG = False;

#by default, all legend texts are set to 'linei'
LEGEND_TEXT = ['ARMA', 'VAR', 'ST-ResNet', 'HA', 'MultiAttConvLSTM']
#LEGEND_TEXT = ['HA','ARMA','VAR','ResNet','MultiAttConvLSTM']
# LEGEND_TEXT[1] = 'line2'
# LEGEND_TEXT[2] = 'line3'

# ================================== plot figure ============================================================
fig = plt.figure()
ax = fig.add_axes([0.15, 0.15, 0.75, 0.75])
#plt.axes([0.15, 0.15, 0.75, 0.75])

#print('POSITION', X_LABEL_POSITION)

x0 = [LEFT_PAD+i*SPACE+(2*i)*haf_group_len for i in range(NUM_GROUP)]
x0 = np.array(x0)
bars =[]
for i in range(NUM_BAR):
    x = x0+i*(BAR_WIDTH+INN_PAD)
    print('x', x)
    bar = plt.bar(x, y[i], width=BAR_WIDTH, color='none', hatch=HATCH_STYLE[i],edgecolor = LINE_COLOR[i],lw=SYMBOL_EDGE_WIDTH)
    plt.bar(x, y[i], width=BAR_WIDTH, label='a', color='none', edgecolor='k', lw=BAR_EDGE_WIDTH)
    bars.append(bar)

# Adjust the subplot layout, because the logit one may take more space
# than usual
# plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
# wspace=0.35)

#plt.text(60, .025, r'$\mu=100,\ \sigma=15$')

# plt.annotate('local max', xy=(2, 1), xytext=(3, 1.5),
#              arrowprops=dict(facecolor='black', shrink=0.05),
#              )

# legend
'''matplotlib.legend.Legend(parent, handles, labels, loc=None, numpoints=None, markerscale=None,
markerfirst=True, scatterpoints=None,
scatteryoffsets=None, prop=None, fontsize=None, borderpad=None,
labelspacing=None, handlelength=None,
handleheight=None, handletextpad=None, borderaxespad=None,
columnspacing=None, ncol=1, mode=None, fancybox=None,
shadow=None, title=None, framealpha=None,
edgecolor=None, facecolor=None, bbox_to_anchor=None,
bbox_transform=None, frameon=None, handler_map=None)'''
#plt.legend(bars,LEGEND_TEXT, loc =LEGEND_POSITION, fontsize=FONT_SIZE_LEGEND, ncol =3, bbox_to_anchor=[0.01,0.99] )
lg = plt.legend(bars,LEGEND_TEXT,loc =LEGEND_POSITION, fontsize=FONT_SIZE_LEGEND)
if not LEGEND_BOX_FLAG:
    lg.draw_frame(False)

# Title and label
if LABEL_TITLE != None:
    plt.title(LABEL_TITLE, fontsize=FONT_SIZE_TITLE, fontname='Times New Roman' )

if LABEL_X != None:
    plt.xlabel(LABEL_X, fontsize=FONT_SIZE_LABEL, fontname='Times New Roman')
if LABEL_Y != None:
    plt.ylabel(LABEL_Y, fontsize=FONT_SIZE_LABEL, fontname='Times New Roman')

# grid
#plt.grid(FLAG_GRID)

# axises
if X_LIM != None:
    plt.xlim(X_LIM)
if Y_LIM != None:
    plt.ylim(Y_LIM)
# major ticks
plt.xticks(X_LABEL_POSITION, XSTICK_LABEL)
# plt.yticks([0.0, 0.2,0.4,0.6],
# 			['-5',  '-3',  '-1',  '1'])

# minor ticks
#ax = plt.gca()
ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(4))
ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(4))
plt.tick_params(which='minor', length=2, color='k')

ax = plt.gca()
# for tick in ax.xaxis.get_major_ticks():
#     tick.label1.set_fontsize(FONT_SIZE_AXIS)
#     tick.label1.set_fontname('Times New Roman')
# for tick in ax.yaxis.get_major_ticks():
#     tick.label1.set_fontsize(FONT_SIZE_AXIS)
#     tick.label1.set_fontname('Times New Roman')

for label in ax.xaxis.get_ticklabels():
    #label.set_color('red')
    #label.set_rotation(45)
    label.set_fontsize(FONT_SIZE_AXIS)
    label.set_fontname('Times New Roman')
for label in ax.yaxis.get_ticklabels():
    #label.set_color('red')
    #label.set_rotation(45)
    label.set_fontsize(FONT_SIZE_AXIS)
    label.set_fontname('Times New Roman')

ax.spines['bottom'].set_linewidth(LINE_WIDTH_AXIS)
ax.spines['left'].set_linewidth(LINE_WIDTH_AXIS)
ax.spines['top'].set_linewidth(LINE_WIDTH_AXIS)
ax.spines['right'].set_linewidth(LINE_WIDTH_AXIS)
for line in ax.xaxis.get_ticklines():
    # line is a Line2D instance
    #line.set_color('green')
    line.set_markersize(3)  # line length
    line.set_markeredgewidth(1.5) # line width
for line in ax.yaxis.get_ticklines():
    # line is a Line2D instance
    #line.set_color('green')     # line color
    line.set_markersize(3)      # line length
    line.set_markeredgewidth(1.5) # line width


#tight_layout automatically adjusts subplot params so that the subplot(s) fits in to the figure area
#plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

foo_fig = plt.gcf()
foo_fig.savefig(figureName)
plt.show()
