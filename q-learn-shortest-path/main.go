package main

import (
	"fmt"
	"github.com/benjohns1/neural-net-go/matutil"
	"gonum.org/v1/gonum/mat"
	"log"
	"math/rand"
	"q-learn-shortest-path/matprint"
	"strings"
	"time"
)

var nanoTime = time.Now().UnixNano()
var seed int64 = nanoTime

// Interesting Seed Cfgs:
// 1628457820267243900 - 5x5 with 500 exp - No path available
// 1628458265665558500 long way around
// 1628458769840143200 20x20 10000 exp - no paths

const (
	envRows = 50
	envCols = 50
	explorationRounds = 300000
	maxIterations = 10000
	startX = 24
	startY = 24
	debug = false
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
	Goal State
}

func (e Environment) String() string {
	var str string
	str += fmt.Sprintf("Tiles:\n%s\n", printer.Table(e.Tiles))
	str += fmt.Sprintf("Rewards:\n%s", printer.Table(e.Rewards))
	return str
}

// LocationKey converts an X, Y coordinate tile value into a location key.
func (e Environment) LocationKey(i, j int) LocationKey {
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

// String for display.
func (key LocationKey) String(e Environment) string {
	x, y := key.Coords(e)
	if debug {
		return fmt.Sprintf("%d(%d,%d)", key, x, y)
	}
	return fmt.Sprintf("(%d,%d)", x, y)
}

// State contains the current state within the environment.
type State struct {
	AgentLoc LocationKey
}

// Action ...
type Action struct {
	Env      Environment
	Location LocationKey
	Reward   float64
}

// NewAction ...
func NewAction(e Environment, loc LocationKey, reward float64) Action {
	return Action{e, loc, reward}
}

func (a Action) String() string {
	return fmt.Sprintf("%s reward: %.f", a.Location.String(a.Env), a.Reward)
}

// ActionsString for display.
func ActionsString(loc LocationKey, e Environment, actions []Action) string {
	str := fmt.Sprintf("Available actions from %s:\n", loc.String(e))
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
		actions = append(actions, NewAction(e, LocationKey(i), v))
	}
	return actions
}

// RandLocation returns a random location in the environment.
func (e Environment) RandLocation() LocationKey {
	r, _ := e.Rewards.Dims()
	return LocationKey(rand.Intn(r))
}

// SampleRandomAction chooses a random action from the list.
func SampleRandomAction(actions []Action) (Action, error) {
	if len(actions) == 0 {
		return Action{}, fmt.Errorf("action list is empty")
	}
	r := rand.Float64()
	step := 1.0 / float64(len(actions))
	curr := step
	for _, action := range actions {
		if r <= curr {
			return action, nil
		}
		curr += step
	}
	return actions[len(actions)-1], nil
}

// QTable for storing values.
type QTable struct {
	Values *mat.Dense
}

// Update q-table given the current state if the agent took action.
//
// Uses the temporal difference method, if future rewards are tied, choose at random.
// 1. Get the current reward, given the current action (from Environment's rewards)
// 2. Find maximum expected value of next action, given current action taken (from Q-table)
// 3. Adjust future max by discount factor
// 4. Add both values together and update Q-value for this state
func (q *QTable) Update(s State, a Action, gamma float64) {
	currentReward := a.Reward
	maxNextQ := q.GetMaxQValue(a.Location)
	updatedQ := currentReward + gamma*maxNextQ
	MatrixSetAtLocations(q.Values, s.AgentLoc, a.Location, updatedQ)
}

// GetMaxQValue gets the max q value for the given location and the available action(s) that yield that value.
func (q QTable) GetMaxQValue(loc LocationKey) float64 {
	allActions := q.Values.RowView(int(loc))
	r, _ := allActions.Dims()
	var maxQ float64
	for i := 0; i < r; i++ {
		qVal := allActions.AtVec(i)
		if qVal > maxQ {
			maxQ = qVal
		}
	}
	return maxQ
}

// GetBestActions filters the list of actions down to the ones that provide the maximum Q value for the given state.
func (q QTable) GetBestActions(state State, availableActions []Action) (float64, []Action) {
	qRow := q.Values.RowView(int(state.AgentLoc))
	c := len(availableActions)
	var maxQ float64
	actions := make([]Action, 0, c)
	for _, potential := range availableActions {
		loc := potential.Location
		qVal := qRow.AtVec(int(loc))
		if qVal == maxQ {
			actions = append(actions, potential)
			continue
		}
		if qVal > maxQ {
			maxQ = qVal
			actions = []Action{potential}
		}
	}
	return maxQ, actions
}

var printer = matprint.NewTableCfg(matprint.Precision(0))

func run() error {
	fmt.Printf("Seed: %d\n", seed)
	env, err := createEnvironment(envRows, envCols)
	if err != nil {
		return err
	}

	qt, err := createQTable(env)
	if err != nil {
		return err
	}

	gamma := 0.8

	if debug {
		fmt.Printf("Rewards:\n%s", printer.Table(env.Rewards))
	}

	fmt.Print("Randomly exploring environment...")
	startExplore := time.Now()
	for i := 0; i < explorationRounds; i++ {
		if debug {
			fmt.Printf("---\nRandom explore round %d:\n", i)
		}
		randomExplore(&env, &qt, gamma)
	}
	endExplore := time.Since(startExplore)

	if debug {
		fmt.Printf("Q-Table:\n%s", printer.Table(qt.Values))
		fmt.Printf("Seed: %d\n", seed)
	}
	fmt.Printf("Tiles:\n%s", printer.Table(env.Tiles))

	fmt.Print("Searching for best path...")
	startPath := time.Now()
	startLocation := env.LocationKey(startX, startY)
	steps, err := env.FindShortestPath(startLocation, &qt, gamma)
	if err != nil {
		return fmt.Errorf("finding shortest path: %v", err)
	}
	stepStrs := make([]string, 0, len(steps))
	for _, step := range steps {
		stepStrs = append(stepStrs, step.String(env))
	}
	endPath := time.Since(startPath)
	fmt.Printf("Path: %s\n", strings.Join(stepStrs, " -> "))
	fmt.Printf("Random exploration took took %v, %d rounds.\n", endExplore, explorationRounds)
	fmt.Printf("%d by %d tile grid. Path from %s to %s.\n", envRows, envCols, startLocation.String(env), env.Goal.AgentLoc.String(env))
	fmt.Printf("Finding shortest path took %v, %d steps.\n", endPath, len(stepStrs)-1)

	return nil
}

// FindShortestPath between the start location and end location.
// Use Q-table to step through from starting state, choosing next action with max value (random breaks tie)
//	- result should be optimal path
func (e Environment) FindShortestPath(start LocationKey, qt *QTable, gamma float64) ([]LocationKey, error) {
	state := State{AgentLoc: start}
	steps := []LocationKey{start}
	for i := 0; state != e.Goal; i++ {
		actions := e.GetAvailableActions(state)
		_, best := qt.GetBestActions(state, actions)
		a, err := SampleRandomAction(best)
		if err != nil {
			return steps, fmt.Errorf("sampling random action: %v", err)
		}
		qt.Update(state, a, gamma)
		steps = append(steps, a.Location)
		state = State{AgentLoc: a.Location}
		if i >= maxIterations {
			return steps, fmt.Errorf("couldn't find path within max iterations (%d)\n", maxIterations)
		}
	}
	return steps, nil
}
// Random exploration to populate Q-table
// - start at a random cell
// - getAvailableActions, sampleNextAction, update state
func randomExplore(env *Environment, qt *QTable, gamma float64) {
	loc := env.RandLocation()
	state := State{AgentLoc: loc}
	actions := env.GetAvailableActions(state)
	randomAction, err := SampleRandomAction(actions)
	if err != nil {
		return
	}

	if debug {
		fmt.Print(ActionsString(loc, *env, actions))
		fmt.Printf("Random action: %s\n", randomAction)
		fmt.Printf("Pre: %.f", qt.Values.At(int(state.AgentLoc), int(randomAction.Location)))
	}
	qt.Update(state, randomAction, gamma)
	if debug {
		fmt.Printf(" Post: %.f\n", qt.Values.At(int(state.AgentLoc), int(randomAction.Location)))
	}
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
			MatrixSetAtLocations(rewards, curr, curr, selfV)
			up, upV := reward(tiles, i-1, j, r, c)
			MatrixSetAtLocations(rewards, curr, up, upV)
			down, downV := reward(tiles, i+1, j, r, c)
			MatrixSetAtLocations(rewards, curr, down, downV)
			left, leftV := reward(tiles, i, j-1, r, c)
			MatrixSetAtLocations(rewards, curr, left, leftV)
			right, rightV := reward(tiles, i, j+1, r, c)
			MatrixSetAtLocations(rewards, curr, right, rightV)
		}
	}

	return rewards, nil
}

// MatrixSetAtLocations ...
func MatrixSetAtLocations(m *mat.Dense, i, j LocationKey, v float64) {
	m.Set(int(i), int(j), v)
}

// Create environment matrix for agent to explore, (-1 = impassable, 0 = passable, 1 = goal)
func createEnvironment(r, c int) (Environment, error) {
	tileValues, goalIndex := randomEnv(r*c)
	tiles, err := matutil.New(r, c, tileValues)
	if err != nil {
		return Environment{} , err
	}

	rewards, err := populateRewardMatrix(tiles)
	if err != nil {
		return Environment{} , err
	}

	return Environment{tiles, rewards, State{AgentLoc: LocationKey(goalIndex)}}, nil
}

func randomEnv(size int) (tiles []float64, goalIndex int) {
	o := make([]float64, size)
	for i := range o {
		v := rand.Float64()
		if v < 0.3 {
			o[i] = -1
		} else {
			o[i] = 0
		}
	}
	goalIndex = int(rand.Float64() * float64(size))
	o[goalIndex] = 1
	return o, goalIndex
}
