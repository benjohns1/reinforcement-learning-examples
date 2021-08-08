package main

import "gonum.org/v1/gonum/mat"

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
