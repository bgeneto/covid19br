# covid19br
This project consists of a single python script that scrapes the web for coronavirus (covid-19) data for Brazil (states and capitals) and process it in order to output .png bar graph files and mp4 (html5, png or gif) animated bar chart race. 

## Installing

Just clone this repository, install the required (prerequisites) packages with your preferred venv tool and execute the python script (see command line options below).

```
cd $HOME
git clone https://github.com/bgeneto/covid19br.git
```

### Prerequisites

This script relies on some python packages, mainly: numpy, matplotlib and pandas. See requirements.txt 
You can install all prerequisites by running the following command:

```
cd $HOME/covid-19
pip3 install -r requirements.txt
```

Additionally, if you want to create HTML5 bar chart racing graphs (-a option), you need to have ffmpeg already installed on your system. A 64-bit binary for Windows is provided in the link below, you have to download it mannualy if using Windows OS and then paste/extract the binary (exe) to same directory as this python script.

[FFmpeg Builds](https://ffmpeg.zeranoe.com/builds/)

OR 

[direct link](https://ffmpeg.zeranoe.com/builds/win64/static/ffmpeg-4.2.2-win64-static.zip)

As always, life is easier on linux, just run your distribution install command (apt, yum etc...) and you are ready to go.

```
sudo apt update && sudo apt install ffmpeg -y
```


## Running the code

The script generates a .ini config file in the first run. You can, as usual, edit this config file to satisfy your needs. 
There is also an input file named 'estados.txt' where you can select all the states you want to scrape info about covid-19 number of cases, number of deaths, cases per million people, and all other info generated by the script. The generated files are outputed to the current script directory in a folder named 'output'. The default input filename can be changed in .ini file or via command line argument (see option -i below).

```
python covid19br.py 
```

OR (to run in parallel and output html5 animation with smoothed transitions)

```
python covid19br.py -s -a html5 -p
```


## Script options

```
usage: covid19br.py [-h] [-v] [-a {gif,html,mp4,png,none}] [-f] [-i INPUT] [-nc] [-ng] [-p] [-s]

This python script scrapes covid-19 data from the web and outputs hundreds of graphs for the selected states in estados.txt file

optional arguments:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -a {gif,html,mp4,png,none}, --anim {gif,html,mp4,png,none}
                        create (html, mp4, png or gif) animated bar racing charts (requires ffmpeg)
  -f, --force           force download of new covid-19 data
  -i INPUT, --input INPUT
                        input file containing the states (takes precedence over ini setting)
  -nc, --no-con         do not check for an active Internet connection
  -ng, --no-graph       do not create any graphs
  -p, --parallel        parallel execution (min. 6 cores, 8GB RAM)
  -s, --smooth          smooth animation transitions by interpolating data
```

NOTE: Use -p or --parallel option with caution. This option will use 6-cores (max) and plenty of memory (8GB or more, depending on state list size).


## License

This project is licensed under the GPL v3 License - see the [LICENSE](LICENSE) file for details
