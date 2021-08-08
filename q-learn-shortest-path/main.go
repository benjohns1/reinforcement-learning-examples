package main

import (
	"fmt"
	"log"

	"github.com/benjohns1/neural-net-go/matutil"
	"gonum.org/v1/gonum/mat"
)

func main() {
	fmt.Println("Q-Learning: Finding the shortest path")
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {

	// todo 1. create environment matrix for agent to explore, (-1 = impassable, 0 = passable, 1 = goal)
	env, err := createEnvironment()
	if err != nil {
		return err
	}
	fmt.Printf("%+v", env)


	// todo:
	//  2. create reward matrix from environment, rows & cols = # of cells in environment, determines reward when
	//     transitioning from one cell to another
	//      - negative reward for transitions to & from impassable cells
	//      - reward for all passable transitions = 0
	//      - reward for moving to goal (even to self) = 100
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

func createEnvironment() (*mat.Dense, error) {
	env, err := matutil.New(5, 5, matutil.FillArray(25, 0))
	if err != nil {
		return nil, err
	}
	return env, nil
}
