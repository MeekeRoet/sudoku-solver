# Sudoku solver

Idea:
1. Create sudoku generator with random agumentations.
2. Train model to recognize four corner points in picture.
3. Deskew image based on corner points and cut up sudoku in 9x9 squares.
5. Recognize which digits are already filled in.
6. Find solution (linear program).
