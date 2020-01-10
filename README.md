# Sudoku solver
This web app allows one to upload - or take and upload, on mobile devices - a photo of a sudoku puzzle and have it
 automatically recognized and solved.

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
