"use client";
import { useState } from "react";
import { Copy, Check } from "lucide-react";

const algorithms = [
  {
    id: 1,
    name: "Prolog",
    code: `P1 - PROLOG

% Define operations
calculate(add, X, Y, Result) :- Result is X + Y.
calculate(subtract, X, Y, Result) :- Result is X - Y.
calculate(multiply, X, Y, Result) :- Result is X * Y.
calculate(divide, X, Y, Result) :- 
    Y \= 0,       % Prevent division by zero
    Result is X / Y.

% Interactive calculator (optional, for command line)
start_calculator :-
    write('Enter first number: '),
    read(X),
    write('Enter operation (add, subtract, multiply, divide): '),
    read(Op),
    write('Enter second number: '),
    read(Y),
    calculate(Op, X, Y, Result),
    format('Result: ~w~n', [Result]).

------------------------------------------------------------------------------------------------------------

% College information
college(jspm_rscoe).

% Departments in the college
department(jspm_rscoe, computer_engineering).
department(jspm_rscoe, information_technology).
department(jspm_rscoe, electronics_telecommunication).
department(jspm_rscoe, mechanical_engineering).
department(jspm_rscoe, civil_engineering).
department(jspm_rscoe, ai_ds).
department(jspm_rscoe, electrical_engineering).

% Example: Courses under departments (optional)
course(computer_engineering, data_structures).
course(computer_engineering, operating_system).
course(information_technology, web_technologies).
course(electronics_telecommunication, signal_processing).

% Example: Faculty members (optional)
faculty(computer_engineering, dr_sharma).
faculty(information_technology, prof_patil).

% Example: Students (optional)
student(jspm_rscoe, sakshi, computer_engineering).
student(jspm_rscoe, rahul, mechanical_engineering).


% Queries example
?- department(jspm_rscoe, Dept).
% Lists all departments in JSPM RSCOE

?- course(computer_engineering, Course).
% Lists courses under Computer Engineering

?- student(jspm_rscoe, Name, Dept).
% Lists all students with their departments

------------------------------------------------------------------------------------------------------------

% -----------------------
% Food Knowledge Base
% -----------------------

% List of all foods
food(apple).
food(banana).
food(carrot).
food(burger).
food(samosa).
food(cucumber).
food(grapes).
food(chips).
food(brinjal).

% Shape of each food item
shape(apple, round).
shape(banana, long).
shape(carrot, cylindrical).
shape(burger, round).
shape(samosa, triangular).
shape(cucumber, cylindrical).
shape(grapes, round).
shape(chips, flat).
shape(brinjal, oval).

% Category of each food item
category(apple, fruit).
category(banana, fruit).
category(carrot, vegetable).
category(burger, snack).
category(samosa, snack).
category(cucumber, vegetable).
category(grapes, fruit).
category(chips, snack).
category(brinjal, vegetable).



% 🔍 Sample Queries to Try:
1. Get all foods with their shape and category:
?- food(F), shape(F, S), category(F, C).

2. Find all fruits that are round:
?- category(F, fruit), shape(F, round).

3. Find shape of a given food:
?- shape(samosa, Shape).

4. Get the category of all cylindrical foods:
?- shape(Food, cylindrical), category(Food, Category).



`,
  },
  {
    id: 2,
    name: "DFS",
    code: `P2 - DFS (NQUEENS)

#include <iostream>
#include <vector>
using namespace std;

class NQueens {
public:
    // Function to solve N-Queens problem
    void solveNQueens(int n) {
        vector<vector<string>> solutions;
        vector<string> board(n, string(n, '.'));
        dfs(0, n, board, solutions);
        
        // Print all solutions
        for (auto& sol : solutions) {
            for (auto& row : sol) {
                cout << row << endl;
            }
            cout << endl;
        }
    }

private:
    // Check if placing a queen at board[row][col] is safe
    bool isSafe(int row, int col, vector<string>& board, int n) {
        // Check column
        for (int i = 0; i < row; i++)
            if (board[i][col] == 'Q')
                return false;

        // Check upper-left diagonal
        for (int i = row - 1, j = col - 1; i >= 0 && j >= 0; i--, j--)
            if (board[i][j] == 'Q')
                return false;

        // Check upper-right diagonal
        for (int i = row - 1, j = col + 1; i >= 0 && j < n; i--, j++)
            if (board[i][j] == 'Q')
                return false;

        return true;
    }

    // DFS to place queens
    void dfs(int row, int n, vector<string>& board, vector<vector<string>>& solutions) {
        if (row == n) {
            solutions.push_back(board); // Found valid arrangement
            return;
        }

        for (int col = 0; col < n; col++) {
            if (isSafe(row, col, board, n)) {
                board[row][col] = 'Q';       // Place queen
                dfs(row + 1, n, board, solutions); // Recurse to next row
                board[row][col] = '.';       // Backtrack
            }
        }
    }
};

int main() {
    int n;
    cout << "Enter number of queens: ";
    cin >> n;

    NQueens nq;
    nq.solveNQueens(n);

    return 0;
}


**Psudo Code:**

function solveNQueens(N):
    board = N x N grid initialized to 0
    DFS(board, 0)

function DFS(board, row):
    if row == N:
        printSolution(board)
        return true

    for col = 0 to N-1:
        if isSafe(board, row, col):
            board[row][col] = 1
            if DFS(board, row + 1):
                return true
            board[row][col] = 0  // backtrack

    return false

function isSafe(board, row, col):
    // check column
    for i in 0 to row-1:
        if board[i][col] == 1:
            return false

    // check upper left diagonal
    for i, j = row-1, col-1 down to 0:
        if board[i][j] == 1:
            return false

    // check upper right diagonal
    for i, j = row-1, col+1 down to 0 and N:
        if board[i][j] == 1:
            return false

    return true

`,
  },
  {
    id: 3,
    name: "BFS",
    code: `P3 - BFS (TSP)

#include <iostream>
#include <vector>
#include <queue>
#include <climits>
using namespace std;

struct State {
    int city;       // current city
    int mask;       // visited cities mask
    int cost;       // total cost so far
};

int tspBFS(vector<vector<int>>& graph) {
    int n = graph.size();
    queue<State> q;

    // Start from city 0, mask with only city 0 visited
    q.push({0, 1, 0});

    int minCost = INT_MAX;

    while (!q.empty()) {
        State cur = q.front();
        q.pop();

        // If all cities are visited
        if (cur.mask == (1 << n) - 1) {
            // Add cost to return to starting city
            minCost = min(minCost, cur.cost + graph[cur.city][0]);
            continue;
        }

        // Try visiting unvisited cities
        for (int next = 0; next < n; next++) {
            if ((cur.mask & (1 << next)) == 0) {  // if not visited
                int newMask = cur.mask | (1 << next);
                int newCost = cur.cost + graph[cur.city][next];
                q.push({next, newMask, newCost});
            }
        }
    }

    return minCost;
}

int main() {
    int n;
    cout << "Enter number of cities: ";
    cin >> n;

    vector<vector<int>> graph(n, vector<int>(n));
    cout << "Enter cost matrix:\n";
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            cin >> graph[i][j];

    int result = tspBFS(graph);
    cout << "Minimum cost to visit all cities and return: " << result << endl;

    return 0;
}
 

Psudo code

function TSP_BFS(graph, start):
    minCost = ∞
    bestPath = []

    queue = Queue()
    queue.enqueue(( [start], 0 ))  // path, cost

    while not queue.isEmpty():
        currentPath, currentCost = queue.dequeue()

        if len(currentPath) == number_of_cities:
            lastCity = currentPath[-1]
            if graph[lastCity][start] != 0:
                totalCost = currentCost + graph[lastCity][start]
                if totalCost < minCost:
                    minCost = totalCost
                    bestPath = currentPath + [start]
            continue

        for nextCity in all cities:
            if nextCity not in currentPath and graph[currentPath[-1]][nextCity] != 0:
                newPath = currentPath + [nextCity]
                newCost = currentCost + graph[currentPath[-1]][nextCity]
                queue.enqueue((newPath, newCost))

    return bestPath, minCost

`,
  },
  {
    id: 4,
    name: "water",
    code: `P4 - WATER JUG

#include <iostream>
#include <queue>
#include <unordered_set>
#include <vector>

using namespace std;

// Function to represent the state of the jugs
struct JugState {
    int jug1, jug2;
    JugState(int j1, int j2) : jug1(j1), jug2(j2) {}
};

// Function to print the steps from start to target
void printSolution(const vector<JugState>& steps) {
    cout << "Steps to reach the target amount of water:" << endl;
    for (const auto& step : steps) {
        cout << "Jug1: " << step.jug1 << ", Jug2: " << step.jug2 << endl;
    }
}

// Function to solve the water jug problem
bool waterJugProblem(int capacity1, int capacity2, int target) {
    if (target > max(capacity1, capacity2)) {
        cout << "Target is greater than both jug capacities. Not possible!" << endl;
        return false;
    }

    unordered_set<string> visited;
    queue<JugState> q;
    vector<JugState> solutionSteps;

    // Starting with both jugs empty
    q.push(JugState(0, 0));
    visited.insert("0_0");

    // Perform BFS to find the solution
    while (!q.empty()) {
        JugState current = q.front();
        q.pop();

        // If we have reached the target amount in either jug
        if (current.jug1 == target || current.jug2 == target) {
            solutionSteps.push_back(current);
            printSolution(solutionSteps);
            return true;
        }

        // Add all possible valid states to the queue
        vector<JugState> nextStates = {
            JugState(capacity1, current.jug2), // Fill Jug 1
            JugState(current.jug1, capacity2), // Fill Jug 2
            JugState(0, current.jug2),         // Empty Jug 1
            JugState(current.jug1, 0),         // Empty Jug 2
            JugState(max(0, current.jug1 - (capacity2 - current.jug2)), min(capacity2, current.jug2 + current.jug1)), // Pour Jug 1 to Jug 2
            JugState(min(capacity1, current.jug1 + current.jug2), max(0, current.jug2 - (capacity1 - current.jug1)))  // Pour Jug 2 to Jug 1
        };

        // For each next state, if it's not visited yet, add it to the queue
        for (const auto& state : nextStates) {
            string stateKey = to_string(state.jug1) + "_" + to_string(state.jug2);
            if (visited.find(stateKey) == visited.end()) {
                visited.insert(stateKey);
                q.push(state);
                solutionSteps.push_back(state);  // Save the path for solution
            }
        }
    }

    cout << "No solution possible!" << endl;
    return false;
}

int main() {
    int capacity1, capacity2, target;

    // Taking user input for jug capacities and target
    cout << "Enter the capacity of Jug 1: ";
    cin >> capacity1;
    cout << "Enter the capacity of Jug 2: ";
    cin >> capacity2;
    cout << "Enter the target amount of water: ";
    cin >> target;

    // Display the input values
    cout << "\nJug 1 capacity = " << capacity1 << endl;
    cout << "Jug 2 capacity = " << capacity2 << endl;
    cout << "Target = " << target << endl;

    // Solving the water jug problem
    if (!waterJugProblem(capacity1, capacity2, target)) {
        cout << "Solution not found!" << endl;
    }

    return 0;
}
`,
  },
  {
    id: 5,
    name: "A* ",
    code: `P5 - A*

#include <iostream>
#include <map>
#include <vector>
#include <set>
#include <queue>
#include <algorithm>

using namespace std;

struct Node {
    string name;
    int g, h, f;
    string parent;

    Node(string n, int g, int h, string p) : name(n), g(g), h(h), f(g + h), parent(p) {}
};

struct Compare {
    bool operator()(const Node& a, const Node& b) {
        return a.f > b.f;
    }
};

void printSet(const string& title, const set<string>& s) {
    cout << title << ": { ";
    for (const auto& item : s) cout << item << " ";
    cout << "}" << endl;
}

int main() {
    map<string, vector<pair<string, int>>> graph;
    map<string, int> h;
    set<string> nodes;
    int edgeCount;

    cout << "Enter number of edges: ";
    cin >> edgeCount;

    cout << "Enter edges in format: from to cost (e.g., A B 4)\n";
    for (int i = 0; i < edgeCount; i++) {
        string from, to;
        int cost;
        cin >> from >> to >> cost;
        graph[from].push_back({to, cost});
        nodes.insert(from);
        nodes.insert(to);
    }

    cout << "\nEnter heuristic (h) for each node:\n";
    for (const auto& node : nodes) {
        int hval;
        cout << "h(" << node << ") = ";
        cin >> hval;
        h[node] = hval;
    }

    string start, goal;
    cout << "\nEnter start node: ";
    cin >> start;
    cout << "Enter goal node: ";
    cin >> goal;

    priority_queue<Node, vector<Node>, Compare> openQueue;
    map<string, int> gCost;
    set<string> closed;
    set<string> openSet;
    map<string, string> parentMap;

    openQueue.push(Node(start, 0, h[start], ""));
    gCost[start] = 0;
    openSet.insert(start);

    cout << "\nStep-by-step A* Execution:\n\n";

    while (!openQueue.empty()) {
        Node current = openQueue.top();
        openQueue.pop();
        openSet.erase(current.name);
        closed.insert(current.name);

        cout << "Current: " << current.name << " (f=" << current.f << ")\n";
        printSet("Open", openSet);
        printSet("Closed", closed);
        cout << "-----------------------\n";

        if (current.name == goal) {
            // Goal found
            vector<string> path;
            string temp = goal;
            while (temp != "") {
                path.push_back(temp);
                temp = parentMap[temp];
            }
            reverse(path.begin(), path.end());

            cout << "\n✅ Path found: ";
            for (const string& node : path) cout << node << " ";
            cout << endl;
            return 0;
        }

        for (auto& neighbor : graph[current.name]) {
            string neighborName = neighbor.first;
            int cost = neighbor.second;
            int newG = current.g + cost;

            if (closed.find(neighborName) != closed.end()) continue;

            if (gCost.find(neighborName) == gCost.end() || newG < gCost[neighborName]) {
                gCost[neighborName] = newG;
                int newH = h[neighborName];
                openQueue.push(Node(neighborName, newG, newH, current.name));
                parentMap[neighborName] = current.name;
                openSet.insert(neighborName);
            }
        }
    }

    cout << "❌ No path found!" << endl;
    return 0;
}
`,
  },
  {
    id: 6,
    name: "AO*",
    code: `P6 - AO*

#include <iostream>
#include <map>
#include <vector>
#include <set>
#include <limits>
#include <algorithm>
#include <climits>
#include <functional> 

using namespace std;

struct EdgeGroup {
    vector<string> nodes;
    int cost;
};

map<string, vector<EdgeGroup>> graph;
map<string, int> heuristic;
map<string, bool> solved;
map<string, EdgeGroup> solutionPath; // ✅ store best edge group for each node
set<string> allNodes;

int aoStar(string node, bool backtrace = false);

void printGraph() {
    cout << "\n📘 Graph:\n";
    for (auto& pair : graph) {
        cout << pair.first << " -> ";
        for (auto& eg : pair.second) {
            cout << "{ ";
            for (auto& n : eg.nodes) cout << n << " ";
            cout << "} cost: " << eg.cost << "  ";
        }
        cout << endl;
    }
}

int aoStar(string node, bool backtrace) {
    if (solved[node]) return heuristic[node];

    int minCost = INT_MAX;
    EdgeGroup bestGroup;

    for (auto& eg : graph[node]) {
        int total = eg.cost;
        for (auto& child : eg.nodes)
            total += heuristic[child];

        if (total < minCost) {
            minCost = total;
            bestGroup = eg;
        }
    }

    heuristic[node] = minCost;
    solutionPath[node] = bestGroup;

    bool allSolved = true;
    for (auto& n : bestGroup.nodes) {
        if (!solved[n]) {
            allSolved = false;
            aoStar(n, true);
        }
    }

    if (allSolved)
        solved[node] = true;

    if (backtrace)
        cout << "🔁 Backtracking to: " << node << " with cost: " << heuristic[node] << endl;

    return heuristic[node];
}

void printSolution(string node) {
    cout << "\n✅ Optimal solution path from root:\n";

    // Print recursively
    function<void(string)> printPath = [&](string current) {
        cout << current << " ";
        if (solutionPath.find(current) != solutionPath.end()) {
            for (auto& child : solutionPath[current].nodes)
                printPath(child);
        }
    };

    printPath(node);
    cout << endl;
}

int main() {
    int edges;
    cout << "Enter number of connections (edges): ";
    cin >> edges;

    cout << "Enter in format -> Parent NumOfChildren Child1 Child2... Cost\n";
    cout << "For OR relation: NumOfChildren = 1, For AND: >1\n";

    set<string> childrenNodes;

    for (int i = 0; i < edges; i++) {
        string parentNode;
        int childCount, cost;
        cin >> parentNode >> childCount;

        EdgeGroup eg;
        for (int j = 0; j < childCount; j++) {
            string child;
            cin >> child;
            eg.nodes.push_back(child);
            childrenNodes.insert(child);
            allNodes.insert(child);
        }
        cin >> cost;
        eg.cost = cost;

        graph[parentNode].push_back(eg);
        allNodes.insert(parentNode);
        solved[parentNode] = false;
    }

    // Ask for h(n) values only for leaf nodes (not in graph keys)
    cout << "\nEnter heuristic values for leaf nodes:\n";
    for (auto& node : allNodes) {
        if (graph.find(node) == graph.end()) {
            int h;
            cout << "h(" << node << ") = ";
            cin >> h;
            heuristic[node] = h;
            solved[node] = true;
        } else {
            heuristic[node] = 0;
        }
    }

    string start;
    cout << "\nEnter start node: ";
    cin >> start;

    printGraph();
    aoStar(start);
    printSolution(start);

    return 0;
}


------------------------------------------------------------------------------------------------------------

/*sample input 1
Enter number of connections (edges): 4
A 2 B C 3
B 1 D 1
C 1 E 2
E 1 F 3

Enter heuristic values for leaf nodes:
h(D) = 2
h(F) = 3

Enter start node: A */

/* sample input 2
Enter number of connections (edges): 5
A 1 B 1
A 2 C D 3
B 1 E 1
C 1 F 2
D 1 G 2

Enter heuristic values for leaf nodes:
h(E) = 2
h(F) = 1
h(G) = 3

Enter start node: A */

`,
  },
  {
    id: 7,
    name: "🙊",
    code: `P7 - MONKEY BANANA

#include <iostream>
using namespace std;

// Structure to represent a state
struct State {
    string monkeyPos;
    string boxPos;
    string bananaPos;
    bool monkeyOnBox;
};

// Function to check if the goal state is reached
bool isGoalState(State s) {
    return (s.monkeyPos == s.boxPos && s.boxPos == s.bananaPos && s.monkeyOnBox);
}

// Function to display current state
void displayState(State s) {
    cout << "Monkey at: " << s.monkeyPos 
         << ", Box at: " << s.boxPos 
         << ", Banana at: " << s.bananaPos 
         << ", Monkey on box: " << (s.monkeyOnBox ? "Yes" : "No") 
         << endl;
}

int main() {
    State currentState;

    // Taking input for initial state
    cout << "Enter Monkey's initial position (e.g. A, B, C): ";
    cin >> currentState.monkeyPos;
    cout << "Enter Box's position (e.g. A, B, C): ";
    cin >> currentState.boxPos;
    cout << "Enter Banana's position (e.g. A, B, C): ";
    cin >> currentState.bananaPos;
    currentState.monkeyOnBox = false;

    cout << "\n--- State Space Simulation ---\n";
    displayState(currentState);

    // Step 1: Move monkey to box
    if (currentState.monkeyPos != currentState.boxPos) {
        cout << "\nAction: Monkey moves to the box.\n";
        currentState.monkeyPos = currentState.boxPos;
        displayState(currentState);
    }

    // Step 2: Push box under banana
    if (currentState.boxPos != currentState.bananaPos) {
        cout << "\nAction: Monkey pushes the box under the banana.\n";
        currentState.boxPos = currentState.bananaPos;
        currentState.monkeyPos = currentState.bananaPos;
        displayState(currentState);
    }

    // Step 3: Climb the box
    cout << "\nAction: Monkey climbs the box.\n";
    currentState.monkeyOnBox = true;
    displayState(currentState);

    // Step 4: Check goal
    if (isGoalState(currentState)) {
        cout << "\n✅ Goal Reached: Monkey grabs the banana! 🍌🎉\n";
    } else {
        cout << "\n❌ Goal Not Reached: Something went wrong.\n";
    }

    return 0;
}

`,
  },
  {
    id: 8,
    name: "world 🗺",
    code: `P8 - BLOCK WORLD


// run using this command - g++ -std=c++17 bw1.cpp -o bw1

----------------------------------------------------------------------------------------------------

#include <iostream>
#include <map>
#include <vector>
#include <string>
#include <algorithm>
#include <set>

using namespace std;

struct BlockState {
    map<string, string> on;       // block -> what it's on
    map<string, bool> clear;      // block -> is clear
    map<string, bool> onTable;    // block -> on table?
    string holding = "";          // currently holding which block
};

void printVisual(BlockState &state) {
    map<string, vector<string>> stacks;
    vector<string> roots;

    for (auto &pair : state.onTable) {
        if (pair.second) roots.push_back(pair.first);
    }

    for (string &root : roots) {
        vector<string> stack;
        string current = root;
        stack.push_back(current);
        while (true) {
            bool found = false;
            for (auto &pair : state.on) {
                if (pair.second == current) {
                    current = pair.first;
                    stack.push_back(current);
                    found = true;
                    break;
                }
            }
            if (!found) break;
        }
        stacks[root] = stack;
    }

    int maxHeight = 0;
    for (auto &s : stacks)
        maxHeight = max(maxHeight, (int)s.second.size());

    for (auto &s : stacks)
        reverse(s.second.begin(), s.second.end());

    cout << "Block World:\n";
    for (int i = 0; i < maxHeight; ++i) {
        for (auto &s : stacks) {
            if (i < s.second.size())
                cout << "| " << s.second[i] << " |\t";
            else
                cout << "      \t";
        }
        cout << endl;
    }
    for (size_t i = 0; i < stacks.size(); ++i)
        cout << "-----\t";
    cout << endl;

    if (state.holding != "")
        cout << "Holding: " << state.holding << endl;

    cout << "---------------------------\n";
}

void pickup(string block, BlockState &state) {
    if (state.onTable[block] && state.clear[block] && state.holding == "") {
        state.holding = block;
        state.onTable[block] = false;
        cout << "Action: PICKUP(" << block << ")\n";
        printVisual(state);
    }
}

void putdown(string block, BlockState &state) {
    if (state.holding == block) {
        state.holding = "";
        state.onTable[block] = true;
        state.clear[block] = true;
        cout << "Action: PUTDOWN(" << block << ")\n";
        printVisual(state);
    }
}

void unstack(string top, BlockState &state) {
    if (state.on.find(top) != state.on.end()) {
        string under = state.on[top];
        if (state.clear[top] && state.holding == "") {
            state.holding = top;
            state.clear[top] = true;
            state.clear[under] = true;
            state.on.erase(top);
            cout << "Action: UNSTACK(" << top << ", " << under << ")\n";
            printVisual(state);
        }
    }
}

void stack(string block, string target, BlockState &state) {
    if (state.clear[target] && state.holding == block) {
        state.holding = "";
        state.on[block] = target;
        state.clear[block] = true;
        state.clear[target] = false;
        cout << "Action: STACK(" << block << ", " << target << ")\n";
        printVisual(state);
    }
}

bool isGoalAchieved(BlockState &current, map<string, string> &goalOn) {
    return current.on == goalOn;
}

void planToGoal(BlockState &current, map<string, string> &goalOn) {
    while (!isGoalAchieved(current, goalOn)) {
        for (auto &[block, goalUnder] : goalOn) {
            string currentUnder = current.on.count(block) ? current.on[block] : (current.onTable[block] ? "table" : "");

            if (goalUnder == "table") {
                if (current.onTable[block]) continue;
                if (!current.clear[block]) {
                    // Unstack top of the block
                    for (auto &[top, under] : current.on) {
                        if (under == block && current.clear[top]) {
                            unstack(top, current);
                            putdown(top, current);
                            break;
                        }
                    }
                }
                unstack(block, current);
                putdown(block, current);
            } else {
                if (current.on.count(block) && current.on[block] == goalUnder) continue;

                // Make sure both blocks are clear
                if (!current.clear[block]) {
                    for (auto &[top, under] : current.on) {
                        if (under == block && current.clear[top]) {
                            unstack(top, current);
                            putdown(top, current);
                            break;
                        }
                    }
                }

                if (!current.clear[goalUnder]) {
                    for (auto &[top, under] : current.on) {
                        if (under == goalUnder && current.clear[top]) {
                            unstack(top, current);
                            putdown(top, current);
                            break;
                        }
                    }
                }

                // Move block to correct position
                if (current.onTable[block]) pickup(block, current);
                else unstack(block, current);

                stack(block, goalUnder, current);
            }

            if (isGoalAchieved(current, goalOn)) break;
        }
    }
}

int main() {
    BlockState current;
    map<string, string> goalOn;

    int n;
    cout << "Enter the number of blocks: ";
    cin >> n;

    vector<string> blocks(n);
    set<string> allBlocks;

    for (int i = 0; i < n; ++i) {
        string name, on;
        cout << "Enter the block name (e.g. A, B, C): ";
        cin >> name;
        blocks[i] = name;
        allBlocks.insert(name);
        cout << "Is it on another block? If yes, enter the block it's on, else enter 'table': ";
        cin >> on;

        if (on == "table") {
            current.onTable[name] = true;
        } else {
            current.on[name] = on;
            current.onTable[name] = false;
            current.clear[on] = false;
        }

        current.clear[name] = true;
    }

    // Fix clear statuses
    for (auto &[top, under] : current.on)
        current.clear[under] = false;

    cout << "Initial State: \n";
    printVisual(current);

    cout << "Enter the goal state:\n";
    for (auto &block : blocks) {
        string target;
        cout << "Enter block name (e.g. A, B, C): " << block << endl;
        cout << "Where should this block be? (Enter 'table' or the block it's on): ";
        cin >> target;

        if (target != "table") {
            goalOn[block] = target;
        }
    }

    cout << "\nPlanning steps to reach goal...\n";
    planToGoal(current, goalOn);

    cout << "Goal state reached.\n";
    return 0;
}

`,
  },

  // Add more if needed
];

export default function PracticalListFixedLeft() {
  const [copiedId, setCopiedId] = useState(null);

  const handleCopy = async (code, id) => {
    await navigator.clipboard.writeText(code);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 1500);
  };

  return (
    <div className="flex w-full min-h-screen bg-gray-50">
      {/* Left sidebar with all practicals */}
      <div className="w-64 h-screen overflow-y-auto sticky top-0 bg-white border-r shadow-sm p-4">
        <ul className="space-y-2 text-[10px] text-gray-700">
          {algorithms.map((algo) => (
            <li key={algo.id} className="flex justify-between items-center">
              <span className="truncate w-44">
                {algo.id}. {algo.name}
              </span>
              <button
                onClick={() => handleCopy(algo.code, algo.id)}
                className="text-gray-500 hover:text-black"
                title="Copy code">
                {copiedId === algo.id ? (
                  <Check size={12} />
                ) : (
                  <Copy size={12} />
                )}
              </button>
            </li>
          ))}
        </ul>
      </div>

      {/* Right side empty or for future use */}
      <div className="flex-1 p-8 flex justify-center items-center text-gray-400 text-sm">
      </div>
    </div>
  );
}
