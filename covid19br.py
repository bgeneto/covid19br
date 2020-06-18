#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
""" Cria gráficos animados com os dados atuais da covid-19 no Brasil.

Esse script busca os dados mais recentes da covid-19 no Brasil e
plota gráficos animados.

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

This program is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
this program. If not, see <http://www.gnu.org/licenses/>.
"""

from urllib.request import Request, urlopen
from pathlib import Path
from datetime import datetime, timedelta
from collections.abc import Iterable
import time
import sys
import shutil
import re
import random
import os
import multiprocessing as mp
import logging
import json
import gzip
import errno
import configparser
import argparse
import matplotlib as mpl
mpl.use('Agg')  # nopep8
import matplotlib.animation as animation
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import requests
import pandas as pd
import numpy as np


__author__ = "bgeneto"
__maintainer__ = "bgeneto"
__contact__ = "bgeneto at gmail"
__copyright__ = "Copyright 2020, bgeneto"
__deprecated__ = False
__license__ = "GPLv3"
__status__ = "Development"
__date__ = "2020/06/17"
__version__ = "0.1.6"


# multiprocessing requires passing these vars explicitly
# use of global vars in problematic in Windows
PARAMS = {
    'SCRIPT_PATH': os.path.dirname(os.path.abspath(sys.argv[0])),
    'SCRIPT_NAME': os.path.splitext(os.path.basename(sys.argv[0]))[0],
    'GTYPES': ['confirmed', 'confirmed_per_mil', 'confirmed_per_den',
               'deaths', 'deaths_per_mil', 'deaths_per_den'
               ],
    'FACECOLOR': '#E5E4E2'
}

LOGGER = logging.getLogger(PARAMS['SCRIPT_NAME'])


def connectionCheck():
    '''
    Simple internet connection checking by using urlopen.
    Returns True (1) on success or False (0) otherwise.
    '''

    # test your internet connection against the following sites:
    TEST_URL = ("google.com", "search.yahoo.com", "bing.com")

    # quick check using urlopen
    for url in TEST_URL:
        try:
            con = urlopen("http://" + url, timeout=10)
            con.read()
            con.close()
            return
        except Exception:
            continue

    # test failed, terminate script execution
    raise ConnectionError(
        "Não foi possível estabelecer uma conexão com a Internet")


def setupLogging(verbose=False):
    """
    Configure script log system
    """
    ch, fh = None, None
    # starts with the highest logging level available
    numloglevel = logging.DEBUG
    LOGGER.setLevel(numloglevel)
    # setup a console logging first
    log2con = int(getIniSetting('log', 'log_to_stdout'))
    if log2con == 1:
        ch = logging.StreamHandler()
        ch.setLevel(numloglevel)
        formatter = logging.Formatter('%(levelname)8s - %(message)s')
        ch.setFormatter(formatter)
        LOGGER.addHandler(ch)

    # now try set log level according to ini file setting
    try:
        loglevel = getIniSetting('log', 'log_level')
        if not verbose:
            numloglevel = getattr(logging, loglevel.upper(), None)
        LOGGER.setLevel(numloglevel)
        if ch:
            ch.setLevel(numloglevel)
    # just in case (safeguard for corrupted .ini file)
    except configparser.NoSectionError:
        createIniFile()
        loglevel = getIniSetting('log', 'log_level')
        if not verbose:
            numloglevel = getattr(logging, loglevel.upper(), None)
        LOGGER.setLevel(numloglevel)
        ch.setLevel(numloglevel)

    # setup file logging
    log2file = int(getIniSetting('log', 'log_to_file'))
    if log2file == 1:
        logfilename = os.path.join(
            PARAMS['SCRIPT_PATH'], "{}.log".format(PARAMS['SCRIPT_NAME']))
        fh = logging.FileHandler(logfilename, mode='w')
        fh.setLevel(numloglevel)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        LOGGER.addHandler(fh)

    return (ch, fh)


def createIniFile(fn=os.path.join(PARAMS['SCRIPT_PATH'], f"{PARAMS['SCRIPT_NAME']}.ini")):
    '''
    Creates an initial config file with default values
    '''
    config = configparser.ConfigParser()
    config.add_section("url")
    config.set("url", "casos",
               "https://data.brasil.io/dataset/covid19/caso.csv.gz")
    config.add_section("log")
    config.set("log", "log_to_file", "1")
    config.set("log", "log_to_stdout", "1")
    config.set("log", "log_level", "info")
    config.add_section("estados")
    config.set("estados", "arquivo", "estados.txt")

    with open(fn, "w", encoding='utf-8') as configFile:
        try:
            config.write(configFile)
        except Exception:
            LOGGER.critical(
                "Não foi possível criar o arquivo .ini de configuração inicial")
            LOGGER.critical("Verifique suas permissões de arquivo")


def getIniConfig():
    '''
    Returns the config object
    '''
    fn = os.path.join(PARAMS['SCRIPT_PATH'], f"{PARAMS['SCRIPT_NAME']}.ini")
    if not os.path.isfile(fn):
        createIniFile(fn)

    config = configparser.ConfigParser()
    config.read(fn)
    return config


def getIniSetting(section, setting):
    '''
    Return a setting value
    '''
    config = getIniConfig()
    value = config.get(section, setting)
    return value


def setupCmdLineArgs():
    """
    Setup script command line arguments
    """
    animate_choices = ["gif", "html", "mp4", "png", "none"]
    parser = argparse.ArgumentParser(
        description='This python script scrapes covid-19 data from the web and outputs hundreds '
                    'of graphs for the selected states in estados.txt file')
    parser.add_argument('-v', '--version', action='version',
                        version=f'%(prog)s v{__version__}')
    parser.add_argument('-a', '--anim', default='mp4', choices=animate_choices,
                        help='create (html, mp4, png or gif) animated bar racing charts (requires ffmpeg)')
    parser.add_argument('-f', '--force', action='store_true', default=False,
                        help='force download of new covid-19 data')
    parser.add_argument('-i', '--input', action='store', default=None,
                        help='input file containing the states (takes precedence over ini setting)')
    parser.add_argument('-nc', '--no-con', action='store_true', default=False, dest="no_con",
                        help='do not check for an active Internet connection')
    parser.add_argument('-ng', '--no-graph', action='store_true', default=False,
                        help='do not create any graphs')
    parser.add_argument('-p', '--parallel', action='store_true', default=False, dest='parallel',
                        help='parallel execution (min. 6 cores, 8GB RAM)')
    parser.add_argument('-s', '--smooth', action='store_true', default=False,
                        help='smooth animation transitions by interpolating data')
    args = parser.parse_args()

    return args


def loadJsonFile(jfn):
    """
    Read a json file and return a corresponding object
    """
    resjson = None
    if os.path.isfile(jfn):
        try:
            with open(jfn, 'r', encoding='utf-8') as fp:
                resjson = json.load(fp)
        except:
            LOGGER.error("Erro ao ler o arquivo '{}'").format(
                os.path.basename(jfn))
    else:
        LOGGER.debug("Arquivo JSON '{}' não localizado".format(jfn))

    return resjson


def getStates():
    '''
    Read states info from csv file and returns a DataFrame
    '''
    # get input file name from ini setting
    arquivo = getIniSetting("estados", "arquivo")
    txtfn = os.path.join(PARAMS['SCRIPT_PATH'], arquivo)
    # command line arguments have priority over ini settings
    if cmdargs.input:
        txtfn = os.path.join(PARAMS['SCRIPT_PATH'], cmdargs.input)
    # csv input file
    csvfn = os.path.join(PARAMS['SCRIPT_PATH'],
                         'input', 'estados-capitais.csv')
    # check if txt file exists and read it
    if os.path.isfile(txtfn):
        with open(txtfn, 'r', encoding='utf-8') as f:
            states = [line.strip().upper() for line in f]
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), txtfn)
    # check if csv input file exists and read it
    if os.path.isfile(csvfn):
        try:
            states_df = pd.read_csv(csvfn)
        except:
            raise ValueError("({})".format(csvfn))
    else:
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), csvfn)

    # remove unwanted states from df
    states_df = states_df[states_df['uf'].isin(states)].reset_index(drop=True)

    return states_df.set_index('uf')


def getIsoDates():
    """
    Returns today and yesterday as formatted strings
    """
    today = datetime.today()
    yesterday = (today - timedelta(1)).date()

    return (today.strftime('%Y-%m-%d'), yesterday.strftime('%Y-%m-%d'))


def downloadCovidData(dt):
    """
    Download total number of covid-19 cases or deaths from the web as csv file
    """

    url = getIniSetting('url', 'casos')
    bname = os.path.basename(url)
    gzfile = os.path.join(PARAMS['SCRIPT_PATH'], "output", "csv",
                          "{}_{}".format(dt, bname))

    if not os.path.isfile(gzfile) or cmdargs.force:
        LOGGER.info("Fazendo o download do arquivo de dados covid-19")
        for _ in range(1, 4):
            try:
                response = requests.get(url, stream=True)
                if response.status_code == 200:
                    with open(gzfile, 'wb') as f:
                        f.write(response.raw.read())
                    break
                else:
                    continue
            except:
                continue
        else:
            raise ConnectionRefusedError(
                errno.ECONNREFUSED, "Falha ao baixar o arquivo csv", bname)

    # extracts gzip file into memory and convert to dataframe
    df = None
    try:
        with gzip.open(gzfile, 'rb') as f_in:
            df = pd.read_csv(f_in)
    except:
        raise Exception(
            "Falha ao extrair o arquivo compactado ('{}')".format(bname))

    return df


def shortDateStr(dt):
    """
    Returns this ugly formated ultra short date string
    Please don't blame me, blame guys at CSSEGISandData
    """
    return dt.strftime("X%m-%e-%y").replace('X0', 'X').replace('X', '')


def setupOutputFolders():
    LOGGER.debug("Criando os diretórios de saída")
    try:
        Path(os.path.join(PARAMS['SCRIPT_PATH'], "output", "csv")).mkdir(
            parents=True, exist_ok=True)
        Path(os.path.join(PARAMS['SCRIPT_PATH'], "output", "png")).mkdir(
            parents=True, exist_ok=True)
    except Exception:
        LOGGER.critical("Não foi possível criar os diretórios necessários")
        LOGGER.critical(
            "Verifique as permissões do sistema de arquivos e tente novamente")
        sys.exit(errno.EROFS)


def swapDate(dt):
    """
    Exchange month with day, day with month
    """
    if isinstance(dt, str):
        try:
            m, d, a = dt.split('/')
            dt = d + '/' + m + '/' + a
        except:
            try:
                m, d, a = dt.split('-')
                dt = d + '-' + m + '-' + a
            except:
                pass

    return dt


def getFlag(code):
    im = None
    fn = os.path.join(PARAMS['SCRIPT_PATH'], 'imagens', code + '.png')
    if os.path.isfile(fn):
        im = plt.imread(fn, format='png')
    else:
        LOGGER.warning("Arquivo '{}' não encontrado".format(
            os.path.basename(fn)))

    return im


def addFlag2Plot(coord, code, zoom, xbox, ax):
    """
    Add a flag image to the plot
    """
    img = getFlag(code)

    if img is None:
        return

    im = OffsetImage(img, zoom=zoom)
    im.image.axes = ax

    ab = AnnotationBbox(im, coord, xybox=(xbox, 0), frameon=False,
                        xycoords='data', boxcoords="offset points", pad=0)

    ax.add_artist(ab)


def setupHbarPlot(vals, y_pos, ylabels, ptype, gtype, dt, ax, color):
    ax.margins(0.20, 0.01)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ylabels, fontsize=14)
    nvals = len(vals)
    # credits
    ax.text(0.985, 0.06, '© 2020 bgeneto', transform=ax.transAxes, ha='right',
            color='#777777', bbox=dict(facecolor='white', alpha=0.75, edgecolor='white'))
    ax.text(0.985, 0.02, 'Fontes: IBGE / Secretarias de Saúde Estaduais / Brasil.IO', transform=ax.transAxes, ha='right',
            color='#777777', bbox=dict(facecolor='white', alpha=0.75, edgecolor='white'))
    fmt = '{:n}'
    if 'per' in gtype:
        fmt = '{:.2f}'
    # add text and flags to every bar
    space = "       "
    zoom = 0.06
    xbox = 14
    if nvals < 12:
        space = "             "
        zoom = 0.12
        xbox = 26
    for code, x, y in zip(vals.index, vals.values, y_pos.values):
        val = fmt.format(round(x, 2))
        pos = int(round(nvals - y + 1))
        ax.text(x, y, space + val + " (P" + str(pos) + ")",
                va='center', ha='left', fontsize=12)
        addFlag2Plot((x, y), code, zoom, xbox, ax)

    tipo = {'city': 'capitais', 'state': 'estados'}
    title = {}
    xlabel = {}
    title['confirmed'] = "Casos de Covid-19 ({}) - {}"
    xlabel['confirmed'] = 'Total de Casos Confirmados'
    title['deaths'] = "Mortes por Covid-19  ({}) - {}"
    xlabel['deaths'] = 'Total de Fatalidades'
    title['confirmed_per_mil'] = "Casos de Covid-19 por Milhão de Habitantes ({})"
    xlabel['confirmed_per_mil'] = 'Total de Casos por Milhão de Habitantes'
    title['deaths_per_mil'] = "Mortes por Covid-19 por Milhão de Habitantes ({})"
    xlabel['deaths_per_mil'] = 'Total de Fatalidades por Milhão de Habitantes'
    title['confirmed_per_den'] = "Casos de Covid-19 por Densidade Demográfica ({})"
    xlabel['confirmed_per_den'] = 'Total de Casos por hab./ha'
    title['deaths_per_den'] = "Mortes por Covid-19 por Densidade Demográfica ({})"
    xlabel['deaths_per_den'] = 'Total de Fatalidades por hab./ha'

    ax.set_xlabel(xlabel[gtype].lower(), fontsize=16)
    ax.set_title(title[gtype].format(
        dt, tipo[ptype]).upper(), fontsize=18, y=1.05)
    ax.xaxis.grid(which='major', alpha=0.5)
    ax.xaxis.grid(which='minor', alpha=0.2)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter('{x:n}'))
    # finally plot
    ax.barh(y_pos, vals, align='center', color=color)


def hbarPlot(df, ptype, gtype, states, cmdargs):
    """
    Horizontal bar plot function
    """

   # change index
    df.set_index('state', inplace=True)

    # latest day
    dt = max(df['date'])

    # store number of rows
    rows = len(df)

    # simple validation
    if rows > len(states):
        LOGGER.error("Número de estados/capitais inválido!")
        return

    # our custom colors
    color_grad = []
    cmap = plt.get_cmap('coolwarm')
    for x in range(rows):
        color_grad.append(cmap(1. * x / rows))

    # auto size w x h
    vsize = round(rows / 3, 2)
    vsize = vsize if vsize > 8 else 8
    fig, ax = plt.subplots(figsize=(vsize * 1.7, vsize))
    vals = df[gtype]
    y_pos = vals.rank(method='first')  # list(range(len(df[gtype])))
    # uggly but required in order to keep original keys order
    codes = list(df.index.values)
    ylabels = [states.loc[v, ptype] for v in codes]
    setupHbarPlot(vals, y_pos, ylabels, ptype, gtype,
                  dt.strftime('%d/%m/%Y'), ax, color_grad)
    ax.set_facecolor(PARAMS['FACECOLOR'])
    fn = os.path.join("output", "png", f"{gtype}_per_{ptype}.png")
    plt.savefig(fn, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close('all')


def animatedPlot(i, df, df_rank, ptype, gtype, states, ax, colors):
    """
    Horizontal bar plot function
    """

    # our ordered subset
    #subdf = df.iloc[i,:].replace(np.nan, 0).sort_values(ascending=True)
    ax.clear()
    ax.xaxis.set_ticks_position('top')
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', colors='#777777', labelsize=11)
    # avoid cutting y-labels (state name)
    plt.gcf().subplots_adjust(left=0.20)
    plt.box(False)
    color = [colors[x] for x in df.columns.tolist()]
    vals = df.iloc[i]
    y_pos = df_rank.iloc[i]
    ylabels = [states.loc[code, ptype] for code in df.columns.values]
    setupHbarPlot(vals, y_pos, ylabels, ptype, gtype,
                  dtFmt2(df.index[i]), ax, color)
    ax.set_xlabel(None)


def linePlot(df, state, states, cmdargs):
    # remove zeroes and nan
    ndf = df.replace(0, np.nan).dropna().sort_values('date', ascending=True)

    gtype = None
    for col in ndf.columns:
        if col in PARAMS['GTYPES']:
            gtype = col
            break

    # graph title
    title = "Número de Casos de Covid-19"
    ylabel = "Total de Casos Confirmados"
    if 'deaths' in gtype:
        title = "Óbitos por Covid-19"
        ylabel = "Total de Fatalidades"

    # plot line color
    color = 'b'
    if 'deaths' in gtype:
        color = 'r'

    # plot size
    hsize = len(ndf) / 9 if len(ndf) / 9 > 8 else 8
    vsize = hsize / 2
    figsize = (hsize, vsize)
    ax = ndf.plot.line(legend=True, color=color,
                       x='date', y=gtype,
                       figsize=figsize)

    # additional visual config
    ax.xaxis_date()
    ax.set_title(title.upper(), fontsize=18)
    ax.set_ylabel(ylabel.lower(), fontsize='large')
    ax.set_xlabel('')
    ax.xaxis.grid(which='major', linestyle='--', alpha=0.5)
    ax.yaxis.grid(which='major', linestyle='--', alpha=0.5)
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.xaxis.set_major_locator(mdates.WeekdayLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%d/%m/%Y'))
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:n}'))
    ax.set_facecolor(PARAMS['FACECOLOR'])
    handles = plt.Rectangle((0, 0), 1, 1, fill=True, color=color)
    ax.legend((handles,), (states.loc[state, 'state'],), loc='upper left',
              frameon=False, shadow=False, fontsize='large')
    ax.text(0.98, 0.10, '© 2020 bgeneto', transform=ax.transAxes, ha='right',
            color='#777777', bbox=dict(facecolor='white', alpha=0.75, edgecolor='white'))
    ax.text(0.98, 0.03, 'Fonte: Secretaria de Saúde / Brasil.IO', transform=ax.transAxes, ha='right',
            color='#777777', bbox=dict(facecolor='white', alpha=0.75, edgecolor='white'))
    plt.xticks(rotation=45)
    fn = os.path.join(PARAMS['SCRIPT_PATH'], "output",
                      "png", f"{state}_{gtype}_historical.png")
    plt.savefig(fn, bbox_inches='tight')
    plt.close('all')


def historicalPlot(df, states, cmdargs):
    """
    Generate historical .png image files for cases and deaths per state
    """
    # plot for selected states only

    for state in states.index:
        if state in df['state'].values:
            if cmdargs.parallel:
                # cases
                pool.apply_async(linePlot, args=(
                    df.loc[df['state'] == state, ['date', 'confirmed']],
                    state, states, cmdargs,)
                )
                # deaths
                pool.apply_async(linePlot, args=(
                    df.loc[df['state'] == state, ['date', 'deaths']],
                    state, states, cmdargs,)
                )
            else:
                linePlot(df.loc[df['state'] == state, [
                    'date', 'confirmed']], state, states, cmdargs)
                linePlot(df.loc[df['state'] == state, [
                    'date', 'deaths']], state, states, cmdargs)
        else:
            LOGGER.error(
                "Estado '{}' não encontrado no arquivo csv ".format(state))


def dtFmt(dt):
    s = str(dt)
    return s[0:4] + '-' + s[4:6] + '-' + s[6:8]


def dtFmt2(dt):
    s = str(dt)
    return s[6:8] + '/' + s[4:6] + '/' + s[0:4]


def createAnimatedGraph(df, ptype, gtype, states, cmdargs):
    """
    Create animated bar racing charts
    """

    # sort dates ascending, removing duplicates
    dates = sorted([int(dt.strftime('%Y%m%d'))
                    for dt in set(df['date'].values)])

    # create a new df to work with
    ndf = pd.DataFrame(data=None, index=dates, columns=states.index.values)
    ndf.index.name = 'date'

    # change date format to int
    df['date'] = df['date'].astype(str).str.replace('-', '').astype(int)
    df.set_index('date', inplace=True)

    # fill new df
    for state in states.index:
        ndf[state] = df.loc[df['state'] == state, gtype]

    # caution: creating/estimating values where values are missing
    # just to get better smoothed transitions
    ndf = ndf.interpolate()

    # smooth transitions
    steps = 5 if cmdargs.smooth else 1
    ndf = ndf.reset_index().replace(np.nan, 0)
    ndf.index = ndf.index * steps
    last_idx = ndf.index[-1] + 1
    ndf = ndf.reindex(range(last_idx))
    ndf['date'] = ndf['date'].fillna(method='ffill')
    ndf = ndf.set_index('date')
    df_rank_expanded = ndf.rank(axis=1, method='first')
    ndf = ndf.interpolate()
    df_rank_expanded = df_rank_expanded.interpolate()

    # animation begins at day
    bday = 25 * steps
    cols = len(states)
    vsize = round(cols / 3 / 1.25, 2)
    vsize = vsize if vsize > 8 else 8
    fig, ax = plt.subplots(figsize=(vsize * 1.77, vsize))

    # color schemes
    random_colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                     for i in range(cols)]
    maximally_dissimilar_colors = [
        '#010067', '#D5FF00', '#FF0056', '#9E008E', '#0E4CA1', '#FFE502', '#005F39', '#00FF00',
        '#95003A', '#FF937E', '#A42400', '#001544', '#91D0CB', '#620E00', '#6B6882', '#0000FF',
        '#007DB5', '#6A826C', '#00AE7E', '#C28C9F', '#BE9970', '#008F9C', '#5FAD4E', '#FF0000',
        '#FF00F6', '#FF029D', '#683D3B', '#FF74A3', '#968AE8', '#98FF52', '#A75740', '#01FFFE',
        '#FE8900', '#BDC6FF', '#01D0FF', '#BB8800', '#7544B1', '#A5FFD2', '#FFA6FE', '#774D00',
        '#7A4782', '#263400', '#004754', '#43002C', '#B500FF', '#FFB167', '#FFDB66', '#90FB92',
        '#7E2DD2', '#BDD393', '#E56FFE', '#DEFF74', '#00FF78', '#009BFF', '#006401', '#0076FF',
        '#85A900', '#00B917', '#788231', '#00FFC6', '#FF6E41', '#E85EBE'
    ]

    # choose your preferred color scheme:
    color_scheme = maximally_dissimilar_colors

    # then we randomize it
    color_lst = random.sample(color_scheme, cols)

    # transform colors to dictionary
    colors = dict(zip(states.index, color_lst))

    # call animator function
    animator = animation.FuncAnimation(fig, animatedPlot, frames=range(bday, len(ndf)),
                                       fargs=(ndf, df_rank_expanded, ptype, gtype, states,
                                              ax, colors),
                                       repeat=False, interval=int(round(1000 / steps)))

    fn = os.path.join(PARAMS['SCRIPT_PATH'], "output",
                      f"{gtype}_animated_{ptype}.{cmdargs.anim}")
    try:
        if cmdargs.anim == "html":
            with open(fn, "w", encoding='utf-8') as html:
                print(animator.to_html5_video(), file=html)
        elif cmdargs.anim == "mp4":
            writer = animation.FFMpegWriter(fps=steps)
            animator.save(fn, writer=writer)
        elif cmdargs.anim == "gif":
            writer = animation.PillowWriter(fps=steps)
            animator.save(fn, writer=writer, savefig_kwargs={
                          'facecolor': PARAMS['FACECOLOR']})
        elif cmdargs.anim == "png":
            from numpngw import AnimatedPNGWriter
            writer = AnimatedPNGWriter(fps=steps)
            animator.save(fn, writer=writer, savefig_kwargs={
                          'facecolor': PARAMS['FACECOLOR']})
    except ModuleNotFoundError:
        raise Exception(
            """Pacote 'numpngw' não disponível! Favor instalar o pacote e tentar posteriormente
           Dica: pip3 install numpngw""")
    except IndexError:
        raise Exception(
            """Pacote 'Pillow' não disponível! Favor instalar o pacote e tentar posteriormente
           Dica: pip3 install Pillow""")
    except (FileNotFoundError, RuntimeError):
        raise Exception(
            """O software 'ffmpeg' não foi localizado! Favor instalar o programa e tentar posteriormente
           Dica: sudo apt update && sudo apt install ffmpeg -y""")


def fmtDataFrameFromCsv(df, sdf):
    """
    DataFrame cleanup
    """
    # convert corresponding column to date
    df['date'] = pd.to_datetime(
        df['date'], format='%Y-%m-%d', errors='coerce').dt.date
    # remove unwanted columns
    df.drop(['city_ibge_code', 'confirmed_per_100k_inhabitants',
             'death_rate'], axis=1, inplace=True)
    # remove unwanted rows:
    # drop states
    df.drop(df[~df['state'].isin(sdf.index.values)].index,
            axis=0, inplace=True)
    # drop cities
    df.drop(df[~(df['city'].isnull()) & ~(
        df['city'].isin(sdf['city']))].index, axis=0, inplace=True)
    # drop cities not in the right state
    for uf in sdf.index:
        df.drop(df[(df['state'] == uf) & (~(df['city'].isnull()) & (df['city'] != sdf.loc[uf, 'city']))].index,
                axis=0, inplace=True)


def hbarSubDf(df, ptype, gtype, last=True):
    if last:
        ret = df.loc[
            (
                (df['is_last']) & (df['place_type'] == ptype)
            ), ['date', 'state', gtype]].sort_values(gtype)
    else:
        ret = df.loc[
            (
                (df['place_type'] == ptype)
            ), ['date', 'state', gtype]].sort_values(gtype)

    return ret


def main():
    # we first confirm that your have an active internet connection
    if not cmdargs.no_con:
        LOGGER.info("Verificando sua conexão com a Internet")
        try:
            connectionCheck()
        except:
            raise

    # get list of state codes from user input text file
    try:
        states_df = getStates()
    except FileNotFoundError:
        raise
    except ValueError:
        LOGGER.critical("Erro ao ler o arquivo csv")
        raise

    today, _ = getIsoDates()

    # download historical covid data for Brazil
    try:
        df = downloadCovidData(today)
    except:
        raise

    # convert to csv to dataframes in order to plot
    fmtDataFrameFromCsv(df, states_df)

    # historical plots of cases and deaths for selected states only
    if not cmdargs.no_graph:
        LOGGER.info("Gerando gráficos png para cada estado/capital")
        historicalPlot(df.loc[df['place_type'] == 'state'],
                       states_df, cmdargs)

    # calculate per mil rates and per (population) density rates
    df['confirmed_per_mil'] = 1e6 * df['confirmed'] / \
        df['estimated_population_2019']
    df['deaths_per_mil'] = 1e6 * df['deaths'] / df['estimated_population_2019']

    # add areas to df
    df.loc[(df["city"].isnull()), 'area'] = df['state'].map(states_df['area'])
    df.loc[(df["city"].isin(states_df['city'])),
           'area'] = df['state'].map(states_df['area_capital'])

    # per population density in acres (ha = hectares)
    df['confirmed_per_den'] = df['area'] * df['confirmed_per_mil'] / 1e4
    df['deaths_per_den'] = df['area'] * df['deaths_per_mil'] / 1e4

    # plot hbars
    if not cmdargs.no_graph:
        LOGGER.info(
            "Favor aguardar, gerando gráficos de barra para cada estado/capital")
        for gtype in PARAMS['GTYPES']:
            for ptype in ['state', 'city']:
                if cmdargs.parallel:
                    pool.apply_async(hbarPlot, args=(
                        hbarSubDf(df, ptype, gtype), ptype, gtype,
                        states_df, cmdargs,)
                    )
                else:
                    hbarPlot(hbarSubDf(df, ptype, gtype), ptype, gtype,
                             states_df, cmdargs)

    # create animated bar graph racing chart
    results = []
    if 'none' not in cmdargs.anim:
        LOGGER.info("Favor aguardar, criando animações com gráficos de barras")
        LOGGER.info("Isso pode levar alguns minutos")
        for gtype in PARAMS['GTYPES']:
            for ptype in ['state', 'city']:
                try:
                    if cmdargs.parallel:
                        results.append(
                            pool.apply_async(createAnimatedGraph, args=(
                                hbarSubDf(df, ptype, gtype,
                                          False), ptype, gtype,
                                states_df, cmdargs,)
                            )
                        )
                    else:
                        LOGGER.info("Considere o uso da opção -p")
                        createAnimatedGraph(
                            hbarSubDf(df, ptype, gtype, False), ptype, gtype, states_df, cmdargs)
                except:
                    raise

    # this is required in order to catch apply_async worker exceptions :-(
    if cmdargs.parallel:
        for result in results:
            result.get()


if __name__ == '__main__':
    # setup logging system
    setupLogging()

    # setup command line arguments
    cmdargs = setupCmdLineArgs()

    # create output directories
    setupOutputFolders()

    pool = None
    # enable parallel exec
    if cmdargs.parallel:
        numcores = int(mp.cpu_count() / 2)
        numthreads = 2 * len(PARAMS['GTYPES']) if numcores > len(
            PARAMS['GTYPES']) else numcores
        pool = mp.Pool(numthreads)

    try:
        main()
        LOGGER.info("Script concluído com sucesso")
    except KeyboardInterrupt:
        LOGGER.critical("Script interrompido pelo usuário")
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    except BaseException as e:
        LOGGER.critical(str(e))
    finally:
        if cmdargs.parallel:
            pool.close()
            pool.join()
