#   Author: Ana Luisa Mata Sanchez
#   Course: CS2302
#   Assignment: Lab #7
#   Instructor: Olac Fuentes
#   Description: Program to draw and solve mazes using DFS and BFS
#   T.A.: Anindita Nath, Maliheh Zargaran
#   Last modified: 04/29/2019
#   Purpose: To compare differences between algorithms

import matplotlib.pyplot as plt
import numpy as np
import random
import time

###################################### Code provided and written by Dr. Fuentes ######################################
def DisjointSetForest(size):
    return np.zeros(size,dtype=np.int)-1
        
def dsfToSetList(S):
    #Returns aa list containing the sets encoded in S
    sets = [ [] for i in range(len(S)) ]
    for i in range(len(S)):
        sets[find(S,i)].append(i)
    sets = [x for x in sets if x != []]
    return sets

def find(S,i):
    # Returns root of tree that i belongs to
    if S[i]<0:
        return i
    return find(S,S[i])

def find_c(S,i): #Find with path compression 
    if S[i]<0: 
        return i
    r = find_c(S,S[i]) 
    S[i] = r 
    return r

def union(S,i,j):
    # Joins i's tree and j's tree, if they are different
    ri = find(S,i) 
    rj = find(S,j)
    if ri!=rj:
        S[rj] = ri

def union_c(S,i,j):
    # Joins i's tree and j's tree, if they are different
    # Uses path compression
    ri = find_c(S,i) 
    rj = find_c(S,j)
    if ri!=rj:
        S[rj] = ri
         
def union_by_size(S,i,j):
    # if i is a root, S[i] = -number of elements in tree (set)
    # Makes root of smaller tree point to root of larger tree 
    # Uses path compression
    ri = find_c(S,i) 
    rj = find_c(S,j)
    if ri!=rj:
        if S[ri]>S[rj]: # j's tree is larger
            S[rj] += S[ri]
            S[ri] = rj
        else:
            S[ri] += S[rj]
            S[rj] = ri



def wall_list(maze_rows, maze_cols):
    # Creates a list with all the walls in the maze
    w =[]
    for r in range(maze_rows):
        for c in range(maze_cols):
            cell = c + r*maze_cols
            if c!=maze_cols-1:
                w.append([cell,cell+1])
            if r!=maze_rows-1:
                w.append([cell,cell+maze_cols])
    return w

def draw_maze(walls,maze_rows,maze_cols,cell_nums=False):
    fig, ax = plt.subplots()
    for w in walls:
        if w[1]-w[0] ==1: #vertical wall
            x0 = (w[1]%maze_cols)
            x1 = x0
            y0 = (w[1]//maze_cols)
            y1 = y0+1
        else:#horizontal wall
            x0 = (w[0]%maze_cols)
            x1 = x0+1
            y0 = (w[1]//maze_cols)
            y1 = y0  
        ax.plot([x0,x1],[y0,y1],linewidth=1,color='k')
    sx = maze_cols
    sy = maze_rows
    ax.plot([0,0,sx,sx,0],[0,sy,sy,0,0],linewidth=2,color='k')
    if cell_nums:
        for r in range(maze_rows):
            for c in range(maze_cols):
                cell = c + r*maze_cols   
                ax.text((c+.5),(r+.5), str(cell), size=10,
                        ha="center", va="center")
    ax.axis('off') 
    ax.set_aspect(1.0)

###################################### MY CODE ######################################

#Method that removes walls and creates the dsf using the standard union and find methods
def create_standard_dsf_maze(S,walls,numWalls,numCells):
    G =[]
    for i in range(numCells):
        G.append([])
    #If there is only one set it means that all cells are reacheable from any cell
    while len(dsfToSetList(S))>1 and numWalls>0:
        #Finds a wall to remove
        d = random.randint(0,len(walls)-1)
        if find(S,walls[d][0]) != find(S,walls[d][1]):
            #make the elements belong to the same set
            union(S,walls[d][0],walls[d][1])
            G[walls[d][0]].append(walls[d][1])
            G[walls[d][1]].append(walls[d][0])
            #remove the wall
            walls.pop(d)
            numWalls = numWalls -1
    if numWalls>0:
        while(numWalls>0 and len(walls)>0):
            d = random.randint(0,len(walls)-1)
            G[walls[d][0]].append(walls[d][1])
            G[walls[d][1]].append(walls[d][0])
            walls.pop(d)
            numWalls = numWalls -1
    return G        
    
#Method that removes walls and creates the dsf using the union by size and compressed find methods
def create_compressed_dsf_maze(SC,wallsC,numWalls,numCells):
    G =[]
    for i in range(numCells):
        G.append([])
    #If there is only one set it means that all cells are reacheable from any cell    
    while len(dsfToSetList(SC))>1 and numWalls>0:
        #Finds a wall to remove
        dC = random.randint(0,len(wallsC)-1)
        #If the elements that share a wall are not in the same set, remove it
        if find_c(SC,wallsC[dC][0]) != find_c(SC,wallsC[dC][1]):
            #make the elements belong to the same set
            union_by_size(SC,wallsC[dC][0],wallsC[dC][1])
            G[wallsC[dC][0]].append(wallsC[dC][1])
            G[wallsC[dC][1]].append(wallsC[dC][0])
            #remove the wall
            wallsC.pop(dC)
            numWalls = numWalls -1
    if numWalls>0:
        while(numWalls>0 and len(wallsC)>0):
            dC = random.randint(0,len(wallsC)-1)
            G[wallsC[dC][0]].append(wallsC[dC][1])
            G[wallsC[dC][1]].append(wallsC[dC][0])
            wallsC.pop(dC)
            numWalls = numWalls -1
    return G

#Method for Breadth First Search inspired by the given graph search pseudocode
def BFS(G, v):
    prev = np.zeros(len(G),dtype=int)-1
    visited = []
    for i in range(len(G)):
        visited.append(False)
    Q = []
    #Start the queue with the first element
    Q.append(v)
    #Mark first element as visited
    visited[v] = True
    while (Q!=[]):
        #Remove current element from queue
        u = Q.pop(0)
        #Go to connected vertices
        for t in G[u]:
            #If the vertex hasn't been visited
            if(visited[t]==False):
                #Mark as visited
                visited[t] = True
                #Add to path
                prev[t] = u
                #Add to the queue
                Q.append(t)
    #Return path
    return prev

#Method for Depth First Search inspired by the given graph search pseudocode
def DFS(G, v):
    prev = np.zeros(len(G),dtype=int)-1
    visited = []
    for i in range(len(G)):
        visited.append(False)
    S = []
    #Start the stack with the first element
    S.insert(0,v)
    #Mark first element as visited
    visited[v] = True
    while (S!=[]):
        #Remove current element from stack
        u = S.pop(0)
        #Go to connected vertices
        for t in G[u]:
            #If the vertex hasn't been visited
            if(visited[t]==False):
                #Mark as visited
                visited[t] = True
                #Add to path
                prev[t] = u
                #Add to Stack
                S.insert(0,t)
    #Return path
    return prev

#Method for Recursive Depth First Search inspired by the given graph search pseudocode
def RecDFS(G, source):
    #Mark element as visited
    visitedd[source] = True
    #Go to connected vertices
    for t in G[source]:
        #Mark as visited
        if visitedd[t] == False:
            #Add to path
            prevv[t] = source
            #Go to next element
            RecDFS(G, t)

#Testing method that prints a path inspired by the given graph search pseudocode           
def printPath(prev, v):
    if (prev[v] !=-1):
        printPath(prev, prev[v])
        print(" - ")
    print(v)

#Method to draw the graph representation of the maze, edited version of the draw_graph code from Dr. Fuentes
def draw_graph(G,maze_rows,maze_cols):
    fig, ax = plt.subplots(figsize=(maze_rows+(maze_rows//4), maze_cols+(maze_cols//4)))
    n = len(G)
    coords =[]
    for i in range(n):
        #Calculate circle location using the calculations in draw_maze
        x0 = (i%maze_cols)
        y0 = (i//maze_cols)
        coords.append([x0,y0])
    for i in range(n):
        for dest in G[i]:
            ax.plot([coords[i][0],coords[dest][0]],[coords[i][1],coords[dest][1]],
                     linewidth=1,color='k')
    for i in range(n):
        ax.text(coords[i][0],coords[i][1],str(i), size=10,ha="center", va="center",
         bbox=dict(facecolor='w',boxstyle="circle"))
    ax.set_aspect(1.0)
    ax.axis('off') 


#Method to draw the solved bfs maze, edited version of the draw_maze code from Dr. Fuentes    
def draw_maze_bfs (walls,maze_rows,maze_cols,G,cell_nums=False):
    fig, ax = plt.subplots()    
    plt.text(maze_cols//2, -maze_cols//16, "BFS", fontdict=None, withdash=False)
    for w in walls:
        if w[1]-w[0] ==1: #vertical wall
            x0 = (w[1]%maze_cols)
            x1 = x0
            y0 = (w[1]//maze_cols)
            y1 = y0+1
        else:#horizontal wall
            x0 = (w[0]%maze_cols)
            x1 = x0+1
            y0 = (w[1]//maze_cols)
            y1 = y0  
        ax.plot([x0,x1],[y0,y1],linewidth=1,color='k')
    sx = maze_cols
    sy = maze_rows
    ax.plot([0,0,sx,sx,0],[0,sy,sy,0,0],linewidth=2,color='k')
    if cell_nums:
        for r in range(maze_rows):
            for c in range(maze_cols):
                cell = c + r*maze_cols   
                ax.text((c+.5),(r+.5), str(cell), size=10,
                        ha="center", va="center")
    #Get BFS path from 0 to the top-right corner
    iBFST = time.time()
    prev = BFS(G, (maze_rows*maze_cols)-1)
    fBFST = time.time()
    print("Time it took to get BFS path:", fBFST-iBFST)

    #Starting vertex
    v = 0
    #Stop when it reaches the end
    while prev[v] !=-1:
        #If the next element in the path is to the left
        if v == prev[v]+1:
            x = (v%maze_cols)+.5
            y = (v//maze_cols)+.5
            x2 = x-1
            y2 = y
        #If the next element in the path is to the right
        elif v == prev[v]-1:
            x = (v%maze_cols)+.5
            y = (v//maze_cols)+.5
            x2 = x+1
            y2 = y
        #If the next element in the path is down
        elif v == prev[v]+maze_cols:
            x = (v%maze_cols)+.5
            y = (v//maze_cols)+.5
            x2 = x
            y2 = y-1
        #If the next element in the path is up
        else:
            x = (v%maze_cols)+.5
            y = (v//maze_cols)+.5
            x2 = x
            y2 = y+1
        ax.plot([x,x2],[y,y2],linewidth=1,color='r')
        #Move to next element
        v = prev[v]
    ax.axis('off') 
    ax.set_aspect(1.0)
    
def draw_maze_dfs (walls,maze_rows,maze_cols,G,cell_nums=False):
    fig, ax = plt.subplots()    
    plt.text(maze_cols//2, -maze_cols//16, "DFS", fontdict=None, withdash=False)
    for w in walls:
        if w[1]-w[0] ==1: #vertical wall
            x0 = (w[1]%maze_cols)
            x1 = x0
            y0 = (w[1]//maze_cols)
            y1 = y0+1
        else:#horizontal wall
            x0 = (w[0]%maze_cols)
            x1 = x0+1
            y0 = (w[1]//maze_cols)
            y1 = y0  
        ax.plot([x0,x1],[y0,y1],linewidth=1,color='k')
    sx = maze_cols
    sy = maze_rows
    ax.plot([0,0,sx,sx,0],[0,sy,sy,0,0],linewidth=2,color='k')
    if cell_nums:
        for r in range(maze_rows):
            for c in range(maze_cols):
                cell = c + r*maze_cols   
                ax.text((c+.5),(r+.5), str(cell), size=10,
                        ha="center", va="center")
                
    #Get DFS path from 0 to the top-right corner
    iDFST = time.time()
    prev = DFS(G, (maze_rows*maze_cols)-1)
    fDFST = time.time()
    print("Time it took to get DFS path:", fDFST-iDFST)
    
    #Starting vertex
    v = 0
    #Stop when it reaches the end
    while prev[v] !=-1:
        #If the next element in the path is to the left
        if v == prev[v]+1:
            x = (v%maze_cols)+.5
            y = (v//maze_cols)+.5
            x2 = x-1
            y2 = y
        #If the next element in the path is to the right
        elif v == prev[v]-1:
            x = (v%maze_cols)+.5
            y = (v//maze_cols)+.5
            x2 = x+1
            y2 = y
        #If the next element in the path is down
        elif v == prev[v]+maze_cols:
            x = (v%maze_cols)+.5
            y = (v//maze_cols)+.5
            x2 = x
            y2 = y-1
        #If the next element in the path is up
        else:
            x = (v%maze_cols)+.5
            y = (v//maze_cols)+.5
            x2 = x
            y2 = y+1
        ax.plot([x,x2],[y,y2],linewidth=1,color='r')
        #Move to next element
        v = prev[v]
    ax.axis('off') 
    ax.set_aspect(1.0)

def draw_maze_recdfs (walls,maze_rows,maze_cols,G,cell_nums=False):
    fig, ax = plt.subplots()    
    plt.text(maze_cols//2, -maze_cols//16, "Recursive DFS", fontdict=None, withdash=False)
    for w in walls:
        if w[1]-w[0] ==1: #vertical wall
            x0 = (w[1]%maze_cols)
            x1 = x0
            y0 = (w[1]//maze_cols)
            y1 = y0+1
        else:#horizontal wall
            x0 = (w[0]%maze_cols)
            x1 = x0+1
            y0 = (w[1]//maze_cols)
            y1 = y0  
        ax.plot([x0,x1],[y0,y1],linewidth=1,color='k')
    sx = maze_cols
    sy = maze_rows
    ax.plot([0,0,sx,sx,0],[0,sy,sy,0,0],linewidth=2,color='k')
    if cell_nums:
        for r in range(maze_rows):
            for c in range(maze_cols):
                cell = c + r*maze_cols   
                ax.text((c+.5),(r+.5), str(cell), size=10,
                        ha="center", va="center")
    
    #Create global variables for the recursive DFS
    global visitedd
    global prevv
    visitedd = [False]*len(G)
    prevv = np.zeros(len(G),dtype=int)-1
    
    #Get recursive DFS path from 0 to the top-right corner
    iRDFST = time.time()
    RecDFS(G, (maze_rows*maze_cols)-1)
    fRDFST = time.time()
    print("Time it took to get recursive DFS path:", fRDFST-iRDFST)
    
    #Starting vertex
    v = 0
    #Stop when it reaches the end
    while prevv[v] !=-1:
        #If the next element in the path is to the left
        if v == prevv[v]+1:
            x = (v%maze_cols)+.5
            y = (v//maze_cols)+.5
            x2 = x-1
            y2 = y
        #If the next element in the path is to the right
        elif v == prevv[v]-1:
            x = (v%maze_cols)+.5
            y = (v//maze_cols)+.5
            x2 = x+1
            y2 = y
        #If the next element in the path is down
        elif v == prevv[v]+maze_cols:
            x = (v%maze_cols)+.5
            y = (v//maze_cols)+.5
            x2 = x
            y2 = y-1
        #If the next element in the path is up
        else:
            x = (v%maze_cols)+.5
            y = (v//maze_cols)+.5
            x2 = x
            y2 = y+1
        ax.plot([x,x2],[y,y2],linewidth=1,color='r')
        #Move to next element
        v = prevv[v]
    ax.axis('off') 
    ax.set_aspect(1.0)



plt.close("all") 
maze_rows = 35
maze_cols = 35
numCells = maze_rows*maze_cols

#wall list & dsf for standard method
walls = wall_list(maze_rows,maze_cols)
S = DisjointSetForest(maze_rows*maze_cols)

#wall list & dsf for compressed method
wallsC = wall_list(maze_rows,maze_cols)
SC = DisjointSetForest(maze_rows*maze_cols)

#draw initial maze
draw_maze(walls,maze_rows,maze_cols,cell_nums=True)
numWalls = int(input('Enter number of walls\n'))

#Display message
if numWalls<(numCells-1):
    print("A path from source to destination is not guaranteed to exist\n")
elif numWalls==(numCells-1):
    print("There is a unique path from source to destination\n")
else:
    print("There is at least one path from source to destination\n")

print("######## Maze using standard find and union ########\n")

iStandardMazeT = time.time()
G = create_standard_dsf_maze(S,walls,numWalls,numCells)        
fStandardMazeT = time.time()

#Draw graph representation
draw_graph(G,maze_rows,maze_cols)
#Draw resulting maze
draw_maze(walls,maze_rows,maze_cols)
#Draw all three solved mazes
draw_maze_bfs(walls,maze_rows,maze_cols,G)
draw_maze_dfs (walls,maze_rows,maze_cols,G)
draw_maze_recdfs (walls,maze_rows,maze_cols,G)

print("Time it took to create the maze:", fStandardMazeT-iStandardMazeT)
print("Maze row size:", maze_rows)
print("Maze column size:", maze_cols)

print("\n######## Maze using compressed find and union by size ########\n")

iCompressedMazeT = time.time()
Gc = create_compressed_dsf_maze(SC,wallsC,numWalls,numCells)        
fCompressedMazeT = time.time()

#Draw graph representation
draw_graph(Gc,maze_rows,maze_cols)
#Draw resulting maze
draw_maze(wallsC,maze_rows,maze_cols)
#Draw all three solved mazes
draw_maze_bfs(wallsC,maze_rows,maze_cols,Gc)
draw_maze_dfs (wallsC,maze_rows,maze_cols,Gc)
draw_maze_recdfs (wallsC,maze_rows,maze_cols,Gc)

print("Time it took to create the maze:", fCompressedMazeT-iCompressedMazeT)
print("Maze row size:", maze_rows)
print("Maze column size:", maze_cols)
