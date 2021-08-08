package main

import (
	"fmt"
	"log"
	"math/rand"
	"q-learn-shortest-path/matprint"
	"strings"
	"time"

	"github.com/benjohns1/neural-net-go/matutil"
	"gonum.org/v1/gonum/mat"
)

const (
	envRows           = 50
	envCols           = 50
	explorationRounds = 250000
	maxIterations     = 10000
	startX            = 0
	startY            = 0
	debug             = false
)

var (
	nanoTime = time.Now().UnixNano()
	seed     = nanoTime
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
		env.RandomlyExplore(&qt, gamma)
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

func reward(rewards mat.Matrix, i, j, r, c int) (k LocationKey, v float64) {
	k, i, j = key(i, j, r, c)
	val := rewards.At(i, j)
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
	tileValues, goalIndex := randomEnv(r * c)
	tiles, err := matutil.New(r, c, tileValues)
	if err != nil {
		return Environment{}, err
	}

	rewards, err := populateRewardMatrix(tiles)
	if err != nil {
		return Environment{}, err
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
