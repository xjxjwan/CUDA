#include <cuda.h>
#include <iostream>
#include <vector>


int main() {
    
    // parameters
    double C = 0.8;
    double gama = 1.4;
    int case_id = 2;
    int nCellsX = 0;
    int nCellsY = 0;
    double tStart = 0.0;
    double tStop = 0.0;
    double x0 = 0.0, y0 = 0.0;
    double x1 = 0.0, y1 = 0.0;
    double dx = 0.0, dy = 0.0;
    std::vector<std::vector<std::array<double, 4>>> u{};

    // initial data
    if (case_id == 1) { // Quadrant problem

        nCellsX = 400; nCellsY = 400;
        x1 = 1.0; y1 = 1.0;
        dx = (x1 - x0) / nCellsX;
        dy = (y1 - y0) / nCellsY;
        tStop = 0.3;
        u.resize(nCellsX + 4, std::vector<std::array<double, 4>>(nCellsY + 4));  // 4 ghost cells

        for (int i = 2; i < nCellsX + 2; i++) {
            for (int j = 2; j < nCellsY + 2; j++) {

                // get coordinates
                double x = x0 + (i - 1.5) * dx;
                double y = y0 + (j - 1.5) * dy;
                std::array<double, 4> u_ij{};

                if (x >= 0.5 && y >= 0.5) {u_ij = {1.5, 0.0, 0.0, 1.5};}
                if (x < 0.5 && y >= 0.5) {u_ij = {0.5325, 1.206, 0.0, 0.3};}
                if (x < 0.5 && y < 0.5) {u_ij = {0.138, 1.206, 1.206, 0.029};}
                if (x >= 0.5 && y < 0.5) {u_ij = {0.5325, 0.0, 1.206, 0.3};}

                // transform from primitive to conservative
                u[i][j] = prim2cons(u_ij, gama);
            }
        }
    }

    if (case_id == 2) { // Shock-bubble interaction

        nCellsX = 500; nCellsY = 197;
        x1 = 225; y1 = 89;
        double bubble_center_x = 35;
        double bubble_center_y = 0.5 * y1;
        dx = (x1 - x0) / nCellsX;
        dy = (y1 - y0) / nCellsY;
        tStop = 0.3;
        u.resize(nCellsX + 4, std::vector<std::array<double, 4>>(nCellsY + 4));  // 4 ghost cells

        for (int i = 2; i < nCellsX + 2; i++) {
            for (int j = 2; j < nCellsY + 2; j++) {

                // get coordinates
                double x = x0 + (i - 1.5) * dx;
                double y = y0 + (j - 1.5) * dy;
                std::array<double, 4> u_ij{};

                if (x < 5) {  // air left to shock
                    u_ij = {1.7755, 110.63, 0.0, 159060.0};
                } else if (pow(pow(x - bubble_center_x, 2) + pow(y - bubble_center_y, 2), 0.5) <= 25) {  // inside bubble
                    u_ij = {0.214, 0.0, 0.0, 101325.0};
                } else {  // air right to shock
                    u_ij = {1.29, 0.0, 0.0, 101325.0};
                }

                // transform from primitive to conservative
                u[i][j] = prim2cons(u_ij, gama);
            }
        }
    }

    // boundary condition
    setBoundaryCondition(u, nCellsX, nCellsY);

}