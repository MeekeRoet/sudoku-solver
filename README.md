# Sudoku solver
This web app lets one upload - or take and upload, on mobile devices - a photo of a sudoku puzzle and have it
 automatically recognized and solved. It also allows the user to correct the digits if they were not recognized correctly. Co-created with three colleagues and a friend.
 
 It looks as follows:

![Screenshot of the web app](webapp_screenshot.png?raw=true")


## Run locally
1. Install GLPK, which is the optimization suite used as a backend by `pyomo` for the sudoku solver. On Linux you can
 use `apt-get install glpk-utils`, on MacOS run `brew install glpk`. For Windows, see
 the instructions [here](http://winglpk.sourceforge.net). 
   
2. Install the python dependencies and run the app:
    ```shell script
    $ cd web-app
    $ pip install -r requirements.txt
    $ python app.py
    ```

## Run in Docker
```bash
$ cd web-app
$ docker build -t sudokusolver .
$ docker run -p 3000:3000 --name sudokusolver-app sudokusolver
```
The app will be accessible on `0.0.0.0:3000`.
