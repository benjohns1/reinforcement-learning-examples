package main

import (
	"fmt"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

// Environment for reinforcement learning
type Environment struct {
	Tiles   *mat.Dense
	Rewards *mat.Dense
	Goal    State
}

func (e Environment) String() string {
	var str string
	str += fmt.Sprintf("Tiles:\n%s\n", printer.Table(e.Tiles))
	str += fmt.Sprintf("Rewards:\n%s", printer.Table(e.Rewards))
	return str
}

// GetAvailableActions for the agent, given a state from environment (where any reward >= 0) (actions are
// coords of accessible nodes)
func (e Environment) GetAvailableActions(state State) []Action {
	currentStateRow := e.Rewards.RowView(int(state.AgentLoc))
	actions := make([]Action, 0, currentStateRow.Len())
	r, _ := currentStateRow.Dims()
	for i := 0; i < r; i++ {
		v := currentStateRow.AtVec(i)
		if v < 0 {
			continue
		}
		actions = append(actions, NewAction(e, LocationKey(i), v))
	}
	return actions
}

// RandLocation returns a random location in the environment.
func (e Environment) RandLocation() LocationKey {
	r, _ := e.Rewards.Dims()
	return LocationKey(rand.Intn(r))
}

// FindShortestPath between the start location and end location.
// Use Q-table to step through from starting state, choosing next action with max value (random breaks tie)
//	- result should be optimal path
func (e Environment) FindShortestPath(start LocationKey, qt *QTable) ([]LocationKey, error) {
	state := State{AgentLoc: start}
	steps := []LocationKey{start}
	for i := 0; state != e.Goal; i++ {
		actions := e.GetAvailableActions(state)
		_, best := qt.GetBestActions(state, actions)
		a, err := SampleRandomAction(best)
		if err != nil {
			return steps, fmt.Errorf("sampling random action: %v", err)
		}
		steps = append(steps, a.Location)
		state = State{AgentLoc: a.Location}
		if i >= maxIterations {
			return steps, fmt.Errorf("couldn't find path within max iterations (%d)\n", maxIterations)
		}
	}
	return steps, nil
}

// RandomlyExplore to populate Q-table
// - start at a random cell
// - getAvailableActions, sampleNextAction, update state
func (e Environment) RandomlyExplore(qt *QTable, gamma float64) {
	loc := e.RandLocation()
	state := State{AgentLoc: loc}
	actions := e.GetAvailableActions(state)
	randomAction, err := SampleRandomAction(actions)
	if err != nil {
		return
	}

	if debug {
		fmt.Print(ActionsString(loc, e, actions))
		fmt.Printf("Random action: %s\n", randomAction)
		fmt.Printf("Pre: %.f", qt.Values.At(int(state.AgentLoc), int(randomAction.Location)))
	}
	qt.Update(state, randomAction, gamma)
	if debug {
		fmt.Printf(" Post: %.f\n", qt.Values.At(int(state.AgentLoc), int(randomAction.Location)))
	}
}

// LocationKey identifies a location within the tile matrix.
type LocationKey int

func key(i, j, rows, cols int) (key LocationKey, newI, newJ int) {
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
	k := (i * cols) + j
	return LocationKey(k), i, j
}

// Coords convert a location key into the X, Y coordinates in the tile matrix.
func (key LocationKey) Coords(e Environment) (i, j int) {
	_, c := e.Tiles.Dims()
	k := int(key)
	i = k / c
	j = k % c
	return i, j
}

// String for display.
func (key LocationKey) String(e Environment) string {
	x, y := key.Coords(e)
	if debug {
		return fmt.Sprintf("%d(%d,%d)", key, x, y)
	}
	return fmt.Sprintf("(%d,%d)", x, y)
}

// LocationKey converts an X, Y coordinate tile value into a location key.
func (e Environment) LocationKey(i, j int) LocationKey {
	r, c := e.Tiles.Dims()
	k, _, _ := key(i, j, r, c)
	return k
}
