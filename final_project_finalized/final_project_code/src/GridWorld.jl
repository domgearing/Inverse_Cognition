module GridWorld

export State, Action, World, actions, step!, is_terminal, manhattan_distance, is_wall

struct State
    x::Int
    y::Int
end

@enum Action begin
    UP
    DOWN
    LEFT
    RIGHT
end

const actions = (UP, DOWN, LEFT, RIGHT)

struct World
    width::Int
    height::Int
    walls::Vector{State}
    goals::Dict{Symbol,State}
end
World(width::Int, height::Int; walls=State[], goals=Dict{Symbol,State}()) =
    World(width, height, walls, goals)

function in_bounds(w::World, s::State)
    1 ≤ s.x ≤ w.width && 1 ≤ s.y ≤ w.height
end

function is_wall(w::World, s::State)
    any(wall -> wall.x == s.x && wall.y == s.y, w.walls)
end

function step!(w::World, s::State, a::Action)
    new_state = State(s.x, s.y)
    if a == UP
        new_state = State(s.x, s.y + 1)
    elseif a == DOWN
        new_state = State(s.x, s.y - 1)
    elseif a == LEFT
        new_state = State(s.x - 1, s.y)
    elseif a == RIGHT
        new_state = State(s.x + 1, s.y)
    end
    #if invalid move, stay in place
    if !in_bounds(w, new_state) || is_wall(w, new_state)
        return s
    else
        return new_state
    end
end

function is_terminal(w::World, s::State, goal::State)
    s.x == goal.x && s.y == goal.y
end

manhattan_distance(s1::State, s2::State) = abs(s1.x - s2.x) + abs(s1.y - s2.y)

end 