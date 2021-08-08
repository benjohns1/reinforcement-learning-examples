package main

import (
	"fmt"
	"github.com/benjohns1/neural-net-go/matutil"
	"gonum.org/v1/gonum/mat"
	"log"
	"math/rand"
	"q-learn-shortest-path/matprint"
	"time"
)

var nanoTime = time.Now().UnixNano()
var seed int64 = 1

const (
	envRows = 4
	envCols = 5
)

func init() {
	rand.Seed(seed)
}

func main() {
	fmt.Println("Q-Learning: Finding the shortest path")
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

// Environment for reinforcement learning
type Environment struct {
	Tiles *mat.Dense
	Rewards *mat.Dense
}

func (e Environment) String() string {
	var str string
	str += fmt.Sprintf("Tiles:\n%s\n", printer.Table(e.Tiles))
	str += fmt.Sprintf("Rewards:\n%s", printer.Table(e.Rewards))
	return str
}

// Key converts an X, Y coordinate tile value into a location key.
func (e Environment) Key(i, j int) LocationKey {
	r, c := e.Tiles.Dims()
	k, _, _ := key(i, j, r, c)
	return k
}

// LocationKey identifies a location within the tile matrix.
type LocationKey int

// Coords convert a location key into the X, Y coordinates in the tile matrix.
func (key LocationKey) Coords(e Environment) (i,j int) {
	_, c := e.Tiles.Dims()
	k := int(key)
	i = k / c
	j = k % c
	return i, j
}

// LocationKeyString for display.
func (key LocationKey) LocationKeyString(e Environment) string {
	x, y := key.Coords(e)
	return fmt.Sprintf("%d(%d,%d)", key, x, y)
}

// State contains the current state within the environment.
type State struct {
	AgentLoc LocationKey
}

// Action ...
type Action struct {
	env Environment
	key LocationKey
	reward float64
}

func (a Action) String() string {
	return fmt.Sprintf("%s reward: %.f", a.key.LocationKeyString(a.env), a.reward)
}

// ActionsString for display.
func ActionsString(loc LocationKey, e Environment, actions []Action) string {
	str := fmt.Sprintf("Available actions from %s:\n", loc.LocationKeyString(e))
	for _, action := range actions {
		str += fmt.Sprintf(" -> %s\n", action)
	}
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
		actions = append(actions, Action{
			env: e,
			key:    LocationKey(i),
			reward: v,
		})
	}
	return actions
}

// SampleRandomAction chooses a random action from the list.
func SampleRandomAction(actions []Action) Action {
	r := rand.Float64()
	step := 1.0 / float64(len(actions))
	curr := step
	for _, action := range actions {
		if r <= curr {
			return action
		}
		curr += step
	}
	return actions[len(actions)-1]
}

// QTable for storing values.
type QTable struct {
	Values *mat.Dense
}

// Update q-table given the current state if the agent took action.
// Uses the temporal difference method, if future rewards are tied, choose at random.
func (q *QTable) Update(e Environment, s State, a Action, g float64) {

}

var printer = matprint.NewTableCfg(matprint.Precision(0))

func run() error {
	env, err := createEnvironment(envRows, envCols)
	if err != nil {
		return err
	}

	qt, err := createQTable(env)
	if err != nil {
		return err
	}

	x, y := 0, 0
	loc := env.Key(x, y)
	state := State{AgentLoc: loc}
	actions := env.GetAvailableActions(state)
	randomAction := SampleRandomAction(actions)

	gamma := 0.8
	qt.Update(env, state, randomAction, gamma)

	fmt.Printf("Rewards:\n%s", printer.Table(env.Rewards))
	fmt.Printf("Tiles:\n%s", printer.Table(env.Tiles))
	fmt.Println(ActionsString(loc, env, actions))
	fmt.Printf("Random action: %s\n", randomAction)
	fmt.Printf("Q-Table:\n%s", printer.Table(qt.Values))

	// todo:
	//  6. exploration loop to populate Q-table
	//      - start at a random cell
	//      - getAvailableActions, sampleNextAction, update state
	//  7. use Q-table to step through from starting state, choosing next action with max value (random breaks tie)
	//      - result should be optimal path


	return nil
}

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

func reward(env mat.Matrix, i, j, r, c int) (k LocationKey, v float64) {
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

// Create q-table, rows & cols = # of cells in environment, init to all zeros
func createQTable(env Environment) (QTable, error) {
	r, c := env.Rewards.Dims()
	values, err := matutil.New(r, c, matutil.FillArray(r*c, 0))
	if err != nil {
		return QTable{}, err
	}
	return QTable{
		Values: values,
	}, nil
}

// Create reward matrix from environment, rows & cols = # of cells in environment, determines reward when
//     transitioning from one cell to another
//      - negative reward for transitions to & from impassable cells
//      - reward for all passable transitions = 0
//      - reward for moving to goal (even to self) = 100
func populateRewardMatrix(tiles mat.Matrix) (*mat.Dense, error) {
	r, c := tiles.Dims()
	dim := r * c
	rewards, err := matutil.New(dim, dim, matutil.FillArray(dim*dim, -1))
	if err != nil {
		return nil, err
	}
	for i := 0; i < r; i++ {
		for j := 0; j < c; j++ {
			curr, _, _ := key(i, j, r, c)
			_, selfV := reward(tiles, i, j, r, c)
			SetForLocations(rewards, curr, curr, selfV)
			up, upV := reward(tiles, i-1, j, r, c)
			SetForLocations(rewards, curr, up, upV)
			down, downV := reward(tiles, i+1, j, r, c)
			SetForLocations(rewards, curr, down, downV)
			left, leftV := reward(tiles, i, j-1, r, c)
			SetForLocations(rewards, curr, left, leftV)
			right, rightV := reward(tiles, i, j+1, r, c)
			SetForLocations(rewards, curr, right, rightV)
		}
	}

	return rewards, nil
}

// SetForLocations ...
func SetForLocations(m *mat.Dense, i, j LocationKey, v float64) {
	m.Set(int(i), int(j), v)
}

// Create environment matrix for agent to explore, (-1 = impassable, 0 = passable, 1 = goal)
func createEnvironment(r, c int) (Environment, error) {
	tiles, err := matutil.New(r, c, randomEnv(r*c))
	if err != nil {
		return Environment{} , err
	}

	rewards, err := populateRewardMatrix(tiles)
	if err != nil {
		return Environment{} , err
	}

	return Environment{tiles, rewards}, nil
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
