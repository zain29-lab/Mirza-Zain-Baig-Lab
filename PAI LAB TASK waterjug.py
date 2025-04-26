class WaterJugDFS:
    def __init__(self, jug1_capacity, jug2_capacity, target):
        self.jug1_capacity = jug1_capacity
        self.jug2_capacity = jug2_capacity
        self.target = target
        self.visited = set()
        
    def is_valid(self, state):
        return state not in self.visited
    
    def dfs(self, state, path):
        jug1, jug2 = state
        
        if state in self.visited:
            return False
        
        self.visited.add(state)
        path.append(state)
        
        print(f"Current state: {state}")
        
        if jug1 == self.target or jug2 == self.target:
            print("Target reached!")
            return True
        
        possible_moves = [
            (self.jug1_capacity, jug2, "Fill Jug 1"),
            (jug1, self.jug2_capacity, "Fill Jug 2"),
            (0, jug2, "Empty Jug 1"),
            (jug1, 0, "Empty Jug 2"),
            (max(0, jug1 - (self.jug2_capacity - jug2)), min(self.jug2_capacity, jug1 + jug2), "Pour Jug 1 -> Jug 2"),
            (min(self.jug1_capacity, jug1 + jug2), max(0, jug2 - (self.jug1_capacity - jug1)), "Pour Jug 2 -> Jug 1")
        ]
        
        for new_state in possible_moves:
            new_jug1, new_jug2, action = new_state
            new_state_tuple = (new_jug1, new_jug2)
            
            if self.is_valid(new_state_tuple):
                print(f"Action: {action}, New state: {new_state_tuple}")
                if self.dfs(new_state_tuple, path):
                    return True
                
        path.pop()
        return False

    def solve(self):
        path = []
        if self.dfs((0, 0), path):
            print("Solution Path:")
            for step in path:
                print(step)
        else:
            print("No solution found.")


jug1_capacity = 5
jug2_capacity = 4
target = 2

solver = WaterJugDFS(jug1_capacity, jug2_capacity, target)
solver.solve()
