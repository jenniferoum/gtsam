add_executable(ABC_EqF
        main.cpp
        EqF.cpp
        State.cpp
        Input.cpp
        G.cpp
        Direction.cpp
        Measurements.cpp
        utilities.cpp
)

target_link_libraries(ABC_EqF gtsam)

