package main

import (
	"fmt"
	"strings"
	"time"
)

type board struct {
	cells [][]bool
}

var memory = make(map[string]bool)

func newBoard(r, c int) board {
	cells := make([][]bool, c)
	for i := range cells {
		cells[i] = make([]bool, r)
	}
	return board{cells: cells}
}

func (b board) String() string {
	rows := make([]string, len(b.cells))
	for i, row := range b.cells {
		cells := make([]string, len(row))
		for j, v := range row {
			if v {
				cells[j] = "Q"
			} else {
				cells[j] = "_"
			}
		}
		rows[i] = strings.Join(cells, " ")
	}
	return strings.Join(rows, "\n")
}

func (b *board) setQueen(x, y int) bool {
	if b.cells[x][y] {
		return false
	}
	b.cells[x][y] = true
	return true
}

func (b *board) unsetQueen(x, y int) {
	b.cells[x][y] = false
}

func (b *board) placeQueen() {
	for i, row := range b.cells {
		for j := range row {
			if !b.setQueen(i, j) {
				continue
			}
			if isBoardSafe(b) {
				return
			}
			b.unsetQueen(i, j)
		}
	}
}

func isBoardSafe(b *board) (result bool) {
	if useMemoization {
		key := fmt.Sprintf("%s", *b)
		if val, ok := memory[key]; ok {
			return val
		}
		defer func() {
			memory[key] = result
		}()
	}

	// Check rows and columns
	foundCols := make([]bool, len(b.cells[0]))
	for i, row := range b.cells {
		foundRow := false
		for j, v := range row {
			if v {
				if foundRow || foundCols[j] || foundDiagonals(b, i, j) {
					return false
				}
				foundRow = true
				foundCols[j] = true
			}
		}
	}

	return true
}

func foundDiagonals(b *board, i, j int) bool {
	downMax := len(b.cells)-1
	rightMax := len(b.cells[0])-1
	up := i-1
	down := i+1
	left := j-1
	right := j+1
	if up >= 0 && left >= 0 {
		if b.cells[up][left] {
			return true
		}
	}
	if up >= 0 && right <= rightMax {
		if b.cells[up][right] {
			return true
		}
	}
	if down <= downMax && left >= 0 {
		if b.cells[down][left] {
			return true
		}
	}
	if down <= downMax && right <= rightMax {
		if b.cells[down][right] {
			return true
		}
	}
	return false
}

func main() {
	run(0, 0)
	run(1, 0)
	run(2, 0)
	run(0, 1)
	run(0, 2)
}

const (
	r, c, queens = 24, 24, 24
	useMemoization = false
)

func run(x, y int) {
	fmt.Println("Solving 8 queens problem")
	b := newBoard(r, c)
	start := time.Now()
	b.setQueen(x, y)
	for i := 0; i < queens-1; i++ {
		b.placeQueen()
	}
	fmt.Printf("%s\n", b)
	fmt.Printf("took %v\n", time.Since(start))
	fmt.Printf("memory count %d\n",len(memory))
}
