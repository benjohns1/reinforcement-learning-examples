package main

import (
	"fmt"
	"github.com/benjohns1/neural-net-go/matutil"
	"gonum.org/v1/gonum/mat"
	"log"
	"math/rand"
)

func init() {
	rand.Seed(1)
}

func main() {
	fmt.Println("Q-Learning: Finding the shortest path")
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	env, err := createEnvironment()
	if err != nil {
		return err
	}
	fmt.Printf("%+v", env)

	rewards, err := createRewardMatrix(env)
	if err != nil {
		return err
	}
	fmt.Printf("%+v", rewards)

	// todo:
	//  3. set gamma to 0.8
	//  4. create q-table, rows & cols = # of cells in environment, init to all zeros
	//  5. helper functions:
	//      - getAvailableActions(state) from environment (where any reward >= 0) (actions are coords of accessible nodes)
	//      - sampleNextAction(actions) random choice
	//      - update(state, action, gamma) updates q-table value (use temporal difference method, if future rewards are tied, choose at random)
	//  6. exploration loop to populate Q-table
	//      - start at a random cell
	//      - getAvailableActions, sampleNextAction, update state
	//  7. use Q-table to step through from starting state, choosing next action with max value (random breaks tie)
	//      - result should be optimal path


	return nil
}

func key(i, j, rows, cols int) (k, newI, newJ int) {
	k = (i * cols) + j
	dims := rows * cols
	if k < 0 {
		k += dims
	} else if k >= dims {
		k -= dims
	}
	if i < 0 {
		i += rows
	} else if i >= rows {
		i -= rows
	}
	if j < 0 {
		j += cols
	} else if j >= cols {
		j -= cols
	}
	return k, i, j
}

func coords(key int, cols int) (i,j int) {
	i = key % cols
	j = key / cols
	return i, j
}

func reward(env mat.Matrix, i, j, r, c int) (k int, v float64) {
	k, i, j = key(i, j, r, c)
	val := env.At(i, j)
	if val < 0 {
		return k, -1
	}
	if val > 0 {
		return k, 100
	}
	return k, 0
}


// Create reward matrix from environment, rows & cols = # of cells in environment, determines reward when
//     transitioning from one cell to another
//      - negative reward for transitions to & from impassable cells
//      - reward for all passable transitions = 0
//      - reward for moving to goal (even to self) = 100
func createRewardMatrix(env mat.Matrix) (*mat.Dense, error) {
	r, c := env.Dims()
	dim := r * c
	rewards, err := matutil.New(dim, dim, matutil.FillArray(dim*dim, -1))
	if err != nil {
		return nil, err
	}
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			curr, _, _ := key(i, j, r, c)
			_, selfV := reward(env, i, j, r, c)
			rewards.Set(curr, curr, selfV)
			up, upV := reward(env, i-1, j, r, c)
			rewards.Set(curr, up, upV)
			down, downV := reward(env, i+1, j, r, c)
			rewards.Set(curr, down, downV)
			left, leftV := reward(env, i, j-1, r, c)
			rewards.Set(curr, left, leftV)
			right, rightV := reward(env, i, j+1, r, c)
			rewards.Set(curr, right, rightV)
		}
	}

	return rewards, nil
}

// Create environment matrix for agent to explore, (-1 = impassable, 0 = passable, 1 = goal)
func createEnvironment() (*mat.Dense, error) {
	env, err := matutil.New(5, 5, randomEnv(25))
	if err != nil {
		return nil, err
	}
	return env, nil
}

func randomEnv(size int) []float64 {
	o := make([]float64, size)
	for i := range o {
		v := rand.Float64()
		if v < 0.3 {
			o[i] = -1
		} else {
			o[i] = 0
		}
	}
	goalIndex := int(rand.Float64() * float64(size))
	o[goalIndex] = 1
	return o
}
