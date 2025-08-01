#include <algorithm>
#include <array>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <cassert>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <string>
#include <filesystem>
#include <ctime>

namespace fs = std::filesystem;


std::array<double, 4> prim2cons(std::array<double, 4> const& u_ij, const double gama) {

    const double rho = u_ij[0];
    const double u = u_ij[1];
    const double v = u_ij[2];
    const double p = u_ij[3];

    std::array<double, 4> res{};
    res[0] = rho;  // rho
    res[1] = rho * u;  // momx
    res[2] = rho * v;  // momy
    res[3] = p / (gama - 1) + 0.5 * rho * (pow(u, 2) + pow(v, 2));  // E

    return res;
}


std::array<double, 4> cons2prim(std::array<double, 4> const& u_ij, const double gama) {

    const double rho = u_ij[0];
    const double momx = u_ij[1];
    const double momy = u_ij[2];
    const double E = u_ij[3];

    std::array<double, 4> res{};
    res[0] = rho;  // rho
    res[1] = momx / rho;  // u
    res[2] = momy / rho;  // v
    res[3] = (gama - 1) * (E - 0.5 * pow(momx, 2) / rho - 0.5 * pow(momy, 2) / rho);  // p

    return res;
}


double computeTimeStep(const std::vector<std::vector<std::array<double, 4>>>& u,
    const double& C, const double& dx, const double& dy, const double& gama) {

    std::vector<double> a_list;
    for (int i = 2; i < u.size() - 2; i++) {
        for (int j = 2; j < u[i].size() - 2; j++) {
            // IMPORTANT: Transform u from cons to prim
            std::array<double, 4> u_prim = cons2prim(u[i][j], gama);
            double cur_u = u_prim[1];  // u
            double cur_v = u_prim[2];  // v
            double vel = pow(pow(cur_u, 2) + pow(cur_v, 2), 0.5);  // non-negative
            double cur_Cs = pow(gama * u_prim[3] / u_prim[0], 0.5);  // p and rho cannot be negative
            double cur_a = vel + cur_Cs;
            a_list.push_back(cur_a);  // the largest eigenvalue (wave speed)
        }
    }

    // for stability: numerical dependence stencil should contain the largest wave speed
    const auto max_iter = std::max_element(a_list.begin(), a_list.end());
    const double timeStep = C * std::min(dx, dy) / *max_iter;
    return timeStep;
}


double getLimiter(const double& r) {

    // // Minbee
    // if (r <= 0) {return 0.0;}
    // if (r > 0 && r <= 1) {return r;}
    // if (r > 1) {return std::min(1.0, 2.0 / (1 + r));}

    // Superbee
    if (r <= 0.0) {return 0.0;}
    if (r > 0.0 and r <= 0.5) {double res = 2 * r; return res;}
    if (r > 0.5 and r <= 1.0) {return 1.0;}
    if (r > 1.0) {double temp = std::min(r, 2.0 / (1 + r)); return std::min(temp, 2.0);}

    return 0;
}


std::vector<std::array<double, 4>> dataReconstruct(std::array<double, 4> const& u_i0, std::array<double, 4> const& u_i,
    std::array<double, 4> const& u_i1) {

    std::array<double, 4> uBarBackward{};
    std::array<double, 4> uBarForward{};

    for (int k = 0; k < 4; k++) {

        const double q_i0 = u_i0[k];
        const double q_i = u_i[k];
        const double q_i1 = u_i1[k];
        double r = (q_i - q_i0) / (q_i1 - q_i);
        double slope_limiter = getLimiter(r);
        // double slope_limiter = 0.0;

        double delta_left = q_i - q_i0;
        double delta_right = q_i1 - q_i;
        double delta_i = 0.5 * (delta_left + delta_right);

        uBarBackward[k] = q_i - 0.5 * slope_limiter * delta_i;
        uBarForward[k] = q_i + 0.5 * slope_limiter * delta_i;
    }

    std::vector<std::array<double, 4>> res{};
    res.push_back(uBarBackward);
    res.push_back(uBarForward);
    return res;
}


std::vector<std::array<double, 4>> halfTimeStepUpdateX(std::array<double, 4> const& uBarL, std::array<double, 4> const& uBarR,
    const double& dx, const double& dt, const double& gama) {

    // variable substitution
    const double& rhoL = uBarL[0], momxL = uBarL[1], momyL = uBarL[2], EL = uBarL[3];
    const double& rhoR = uBarR[0], momxR = uBarR[1], momyR = uBarR[2], ER = uBarR[3];
    std::array<double, 4> uBarL_prim = cons2prim(uBarL, gama);
    std::array<double, 4> uBarR_prim = cons2prim(uBarR, gama);
    const double& vxL = uBarL_prim[1], vyL = uBarL_prim[2], pL = uBarL_prim[3];
    const double& vxR = uBarL_prim[1], vyR = uBarR_prim[2], pR = uBarR_prim[3];

    // half-time-step update
    // flux functions used
    double rho_update = 0.5 * (dt / dx) * (momxR - momxL);
    double momx_update = 0.5 * (dt / dx) * (rhoR * pow(vxR, 2) + pR - rhoL * pow(vxL, 2) - pL);
    double momy_update = 0.5 * (dt / dx) * (rhoR * vyR * vxR - rhoL * vyL * vxL);
    double E_update = 0.5 * (dt / dx) * ((ER + pR) * vxR - (EL + pL) * vxL);

    double rhoBarLUpdate = rhoL - rho_update, rhoBarRUpdate = rhoR - rho_update;
    double momxBarLUpdate = momxL - momx_update, momxBarRUpdate = momxR - momx_update;
    double momyBarLUpdate = momyL - momy_update, momyBarRUpdate = momyR - momy_update;
    double EBarLUpdate = EL - E_update, EBarRUpdate = ER - E_update;

    std::array<double, 4> uBarLUpdate = {rhoBarLUpdate, momxBarLUpdate, momyBarLUpdate, EBarLUpdate};
    std::array<double, 4> uBarRUpdate = {rhoBarRUpdate, momxBarRUpdate, momyBarRUpdate, EBarRUpdate};

    std::vector<std::array<double, 4>> res{};
    res.push_back(uBarLUpdate);
    res.push_back(uBarRUpdate);
    return res;
}


std::vector<std::array<double, 4>> halfTimeStepUpdateY(std::array<double, 4> const& uBarD, std::array<double, 4> const& uBarU,
    const double& dy, const double& dt, const double& gama) {

    // variable substitution
    const double& rhoD = uBarD[0], momxD = uBarD[1], momyD = uBarD[2], ED = uBarD[3];
    const double& rhoU = uBarU[0], momxU = uBarU[1], momyU = uBarU[2], EU = uBarU[3];
    std::array<double, 4> uBarD_prim = cons2prim(uBarD, gama);
    std::array<double, 4> uBarU_prim = cons2prim(uBarU, gama);
    const double& vxD = uBarD_prim[1], vyD = uBarD_prim[2], pD = uBarD_prim[3];
    const double& vxU = uBarU_prim[1], vyU = uBarU_prim[2], pU = uBarU_prim[3];

    // half-time-step update
    // flux functions used
    double rho_update = 0.5 * (dt / dy) * (momyU - momyD);
    double momx_update = 0.5 * (dt / dy) * (rhoU * vxU * vyU - rhoD * vxD * vyD);
    double momy_update = 0.5 * (dt / dy) * (rhoU * pow(vyU, 2) + pU - rhoD * pow(vyD, 2) - pD);
    double E_update = 0.5 * (dt / dy) * ((EU + pU) * vyU - (ED + pD) * vyD);

    double rhoBarDUpdate = rhoD - rho_update, rhoBarUUpdate = rhoU - rho_update;
    double momxBarDUpdate = momxD - momx_update, momxBarUUpdate = momxU - momx_update;
    double momyBarDUpdate = momyD - momy_update, momyBarUUpdate = momyU - momy_update;
    double EBarDUpdate = ED - E_update, EBarUUpdate = EU - E_update;

    std::array<double, 4> uBarDUpdate = {rhoBarDUpdate, momxBarDUpdate, momyBarDUpdate, EBarDUpdate};
    std::array<double, 4> uBarUUpdate = {rhoBarUUpdate, momxBarUUpdate, momyBarUUpdate, EBarUUpdate};

    std::vector<std::array<double, 4>> res{};
    res.push_back(uBarDUpdate);
    res.push_back(uBarUUpdate);
    return res;
}


std::array<double, 4> getFluxX(std::array<double, 4> const& u_i, std::array<double, 4> const& u_i1, const double& dx, const double& dt, const double& gama) {

    // variable substitution
    const double& rho_i = u_i[0], momx_i = u_i[1], momy_i = u_i[2], E_i = u_i[3];
    const double& rho_i1 = u_i1[0], momx_i1 = u_i1[1], momy_i1 = u_i1[2], E_i1 = u_i1[3];
    std::array<double, 4> u_i_prim = cons2prim(u_i, gama);
    std::array<double, 4> u_i1_prim = cons2prim(u_i1, gama);
    const double& vx_i = u_i_prim[1], vy_i = u_i_prim[2], vx_i1 = u_i1_prim[1], vy_i1 = u_i1_prim[2];
    const double& p_i = u_i_prim[3], p_i1 = u_i1_prim[3];

    // L-F scheme
    const double F_rho_LF = 0.5 * dx / dt * (rho_i - rho_i1) + 0.5 * (momx_i + momx_i1);
    const double F_momx_LF = 0.5 * dx / dt * (momx_i - momx_i1) + 0.5 * (rho_i * pow(vx_i, 2) + p_i + rho_i1 * pow(vx_i1, 2) + p_i1);
    const double F_momy_LF = 0.5 * dx / dt * (momy_i - momy_i1) + 0.5 * (rho_i * vy_i * vx_i + rho_i1 * vy_i1 * vx_i1);
    const double F_E_LF = 0.5 * dx / dt * (E_i - E_i1) + 0.5 * ((E_i + p_i) * vx_i + (E_i1 + p_i1) * vx_i1);
    const std::array F_LF = {F_rho_LF, F_momx_LF, F_momy_LF, F_E_LF};

    // RI scheme
    const double rho_boundary = 0.5 * (rho_i + rho_i1) - 0.5 * dt/dx * (momx_i1 - momx_i);
    const double momx_boundary = 0.5 * (momx_i + momx_i1) - 0.5 * dt/dx * (rho_i1 * pow(vx_i1, 2) + p_i1 - rho_i * pow(vx_i, 2) - p_i);
    const double momy_boundary = 0.5 * (momy_i + momy_i1) - 0.5 * dt/dx * (rho_i1 * vy_i1 * vx_i1 - rho_i * vy_i * vx_i);
    const double E_boundary = 0.5 * (E_i + E_i1) - 0.5 * dt/dx * ((E_i1 + p_i1) * vx_i1 - (E_i + p_i) * vx_i);

    const double vx_boundary = momx_boundary / rho_boundary;
    const double vy_boundary = momy_boundary / rho_boundary;
    const double p_boundary = (gama - 1) * (E_boundary - 0.5 * pow(momx_boundary, 2) / rho_boundary - 0.5 * pow(momy_boundary, 2) / rho_boundary);

    const double F_rho_RI = momx_boundary;
    const double F_momx_RI = rho_boundary * pow(vx_boundary, 2) + p_boundary;
    const double F_momy_RI = rho_boundary * vy_boundary * vx_boundary;
    const double F_E_RI = (E_boundary + p_boundary) * vx_boundary;
    const std::array F_RI = {F_rho_RI, F_momx_RI, F_momy_RI, F_E_RI};

    // FORCE scheme
    const std::array F_FORCE = {0.5 * (F_LF[0] + F_RI[0]), 0.5 * (F_LF[1] + F_RI[1]),
        0.5 * (F_LF[2] + F_RI[2]), 0.5 * (F_LF[3] + F_RI[3])};

    return F_FORCE;
}


std::array<double, 4> getFluxY(std::array<double, 4> const& u_i, std::array<double, 4> const& u_i1, const double& dy, const double& dt, const double& gama) {

    // variable substitution
    const double& rho_i = u_i[0], momx_i = u_i[1], momy_i = u_i[2], E_i = u_i[3];
    const double& rho_i1 = u_i1[0], momx_i1 = u_i1[1], momy_i1 = u_i1[2], E_i1 = u_i1[3];
    std::array<double, 4> u_i_prim = cons2prim(u_i, gama);
    std::array<double, 4> u_i1_prim = cons2prim(u_i1, gama);
    const double& vx_i = u_i_prim[1], vy_i = u_i_prim[2], vx_i1 = u_i1_prim[1], vy_i1 = u_i1_prim[2];
    const double& p_i = u_i_prim[3], p_i1 = u_i1_prim[3];

    // L-F scheme
    const double F_rho_LF = 0.5 * dy / dt * (rho_i - rho_i1) + 0.5 * (momy_i + momy_i1);
    const double F_momx_LF = 0.5 * dy / dt * (momx_i - momx_i1) + 0.5 * (rho_i * vx_i * vy_i + rho_i1 * vx_i1 * vy_i1);
    const double F_momy_LF = 0.5 * dy / dt * (momy_i - momy_i1) + 0.5 * (rho_i * pow(vy_i, 2) + p_i + rho_i1 * pow(vy_i1, 2) + p_i1);
    const double F_E_LF = 0.5 * dy / dt * (E_i - E_i1) + 0.5 * ((E_i + p_i) * vy_i + (E_i1 + p_i1) * vy_i1);
    const std::array F_LF = {F_rho_LF, F_momx_LF, F_momy_LF, F_E_LF};

    // RI scheme
    const double rho_boundary = 0.5 * (rho_i + rho_i1) - 0.5 * dt/dy * (momy_i1 - momy_i);
    const double momx_boundary = 0.5 * (momx_i + momx_i1) - 0.5 * dt/dy * (rho_i1 * vx_i1 * vy_i1 - rho_i * vx_i * vy_i);
    const double momy_boundary = 0.5 * (momy_i + momy_i1) - 0.5 * dt/dy * (rho_i1 * pow(vy_i1, 2) + p_i1 - rho_i * pow(vy_i, 2) - p_i);
    const double E_boundary = 0.5 * (E_i + E_i1) - 0.5 * dt/dy * ((E_i1 + p_i1) * vy_i1 - (E_i + p_i) * vy_i);

    const double vx_boundary = momx_boundary / rho_boundary;
    const double vy_boundary = momy_boundary / rho_boundary;
    const double p_boundary = (gama - 1) * (E_boundary - 0.5 * pow(momx_boundary, 2) / rho_boundary - 0.5 * pow(momy_boundary, 2) / rho_boundary);

    const double F_rho_RI = momy_boundary;
    const double F_momx_RI = rho_boundary * vx_boundary * vy_boundary;
    const double F_momy_RI = rho_boundary * pow(vy_boundary, 2) + p_boundary;
    const double F_E_RI = (E_boundary + p_boundary) * vy_boundary;
    const std::array F_RI = {F_rho_RI, F_momx_RI, F_momy_RI, F_E_RI};

    // FORCE scheme
    const std::array F_FORCE = {0.5 * (F_LF[0] + F_RI[0]), 0.5 * (F_LF[1] + F_RI[1]),
        0.5 * (F_LF[2] + F_RI[2]), 0.5 * (F_LF[3] + F_RI[3])};

    return F_FORCE;
}


void setBoundaryCondition(std::vector<std::vector<std::array<double, 4>>>& u, const int& nCellsX, const int& nCellsY) {

    // transmissive boundary condition
    for (int i = 2; i < nCellsX + 2; i++) {
        u[i][0] = u[i][2];
        u[i][1] = u[i][2];
    }
    for (int i = 2; i < nCellsX + 2; i++) {
        u[i][nCellsY + 2] = u[i][nCellsY + 1];
        u[i][nCellsY + 3] = u[i][nCellsY + 1];
    }
    for (int j = 0; j < nCellsY + 4; j++) {
        u[0][j] = u[2][j];
        u[1][j] = u[2][j];
    }
    for (int j = 0; j < nCellsY + 4; j++) {
        u[nCellsX + 2][j] = u[nCellsX + 1][j];
        u[nCellsX + 3][j] = u[nCellsX + 1][j];
    }
}


void dataRecord(const std::vector<std::vector<std::array<double, 4>>>& u, const int case_id, const double nCellsX,
    const double nCellsY, const double x0, const double y0, const double dx, const double dy,
    const double t, const double gama) {

    // transform
    std::vector<std::vector<std::array<double, 4>>> u_prim;
    u_prim.resize(nCellsX + 4, std::vector<std::array<double, 4>>(nCellsY + 4));
    for (int i = 0; i < nCellsX + 4; i++) {
        for (int j = 0; j < nCellsY + 4; j++) {
            u_prim[i][j] = cons2prim(u[i][j], gama);
        }
    }

    // check whether the directory exists, create one if not
    std::ostringstream folderPath;
    folderPath << "res/Case_" << case_id;
    std::string caseFolder = folderPath.str();
    if (!fs::exists(caseFolder)) {
        fs::create_directories(caseFolder);
    }

    // data recording
    std::ostringstream oss;
    oss << caseFolder << "/T=" << std::setprecision(2) << t << ".txt";
    std::string fileName = oss.str();
    std::fstream outFile(fileName, std::ios::out);
    for (int i = 2; i < nCellsX + 2; i++) {
        for (int j = 2; j < nCellsY + 2; j++) {
            // std::cout << x0 + (i - 1) * dx << ", " << u[i][0] << ", " << u[i][1] << ", " << u[i][2] << std::endl;
            outFile << x0 + (i - 1.5) * dx << ", " << y0 + (j - 1.5) * dy
            << ", " << u_prim[i][j][0] << ", " << u_prim[i][j][1] << ", " << u_prim[i][j][2] << ", " << u_prim[i][j][3]
            << std::endl;
        }
    }
    outFile.close();
}


int main() {

    // experimental options
    int case_id = 2;
    bool Record = false;

    // parameters
    double C = 0.8;
    double gamma = 1.4;
    int nCellsX = 0;
    int nCellsY = 0;
    double tStart = 0.0;
    double tStop = 0.0;
    double x0 = 0.0, y0 = 0.0;
    double x1 = 0.0, y1 = 0.0;
    double dx = 0.0, dy = 0.0;
    double time_ratio = 1.0;  // used for unscaling time in case 2
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
                u[i][j] = prim2cons(u_ij, gamma);
            }
        }
    }

    if (case_id == 2) { // Shock-bubble interaction

        nCellsX = 500; nCellsY = 197;
        x1 = 0.225; y1 = 0.089;
        double bubble_center_x = 0.035;
        double bubble_center_y = 0.0445;
        dx = (x1 - x0) / nCellsX;
        dy = (y1 - y0) / nCellsY;
        double bubble_radius = 0.025;
        double Ms = 1.22, p_Air = 101325.0, rho_Air = 1.29;
        double Cs_Air = pow(gamma * p_Air / rho_Air, 0.5);
        time_ratio = bubble_radius / (Cs_Air * Ms);
        tStop = 19 * time_ratio;
        u.resize(nCellsX + 4, std::vector<std::array<double, 4>>(nCellsY + 4));  // 4 ghost cells

        for (int i = 2; i < nCellsX + 2; i++) {
            for (int j = 2; j < nCellsY + 2; j++) {

                // get coordinates
                double x = x0 + (i - 1.5) * dx;
                double y = y0 + (j - 1.5) * dy;
                std::array<double, 4> u_ij{};

                if (x < 0.005) {  // air left to shock
                    u_ij = {1.7755, 110.63, 0.0, 159060.0};
                } else if (pow(pow(x - bubble_center_x, 2) + pow(y - bubble_center_y, 2), 0.5) <= bubble_radius) {  // inside bubble
                    u_ij = {0.214, 0.0, 0.0, 101325.0};
                } else {  // air right to shock
                    u_ij = {1.29, 0.0, 0.0, 101325.0};
                }

                // transform from primitive to conservative
                u[i][j] = prim2cons(u_ij, gamma);
            }
        }
    }

    // boundary condition
    setBoundaryCondition(u, nCellsX, nCellsY);

    // execution time recording
    double elapsdt = 0, elapsx = 0, elapsy = 0, elapsbc = 0, elapstotal = 0;
    clock_t startx, endx, starty, endy, startdt, enddt, startbc, endbc, start, end;

    // update data
    double t = tStart;
    // std::array record_list = {0.1, 0.2, 0.3};
    std::array record_list = {0.6, 1.2, 1.8, 3.0, 4.6, 6.2, 7.8, 9.4, 12.6, 17.4, 19.0};
    int record_index = 0;
    int counter = 0;
    do {

        start = clock();

        // data storage
        std::vector<std::vector<std::array<double, 4>>> uBarL;
        uBarL.resize(nCellsX + 4, std::vector<std::array<double, 4>>(nCellsY + 4));
        std::vector<std::vector<std::array<double, 4>>> uBarR;
        uBarR.resize(nCellsX + 4, std::vector<std::array<double, 4>>(nCellsY + 4));
        std::vector<std::vector<std::array<double, 4>>> uBarU;
        uBarU.resize(nCellsX + 4, std::vector<std::array<double, 4>>(nCellsY + 4));
        std::vector<std::vector<std::array<double, 4>>> uBarD;
        uBarD.resize(nCellsX + 4, std::vector<std::array<double, 4>>(nCellsY + 4));

        std::vector<std::vector<std::array<double, 4>>> uBarLUpdate;
        uBarLUpdate.resize(nCellsX + 4, std::vector<std::array<double, 4>>(nCellsY + 4));
        std::vector<std::vector<std::array<double, 4>>> uBarRUpdate;
        uBarRUpdate.resize(nCellsX + 4, std::vector<std::array<double, 4>>(nCellsY + 4));
        std::vector<std::vector<std::array<double, 4>>> uBarUUpdate;
        uBarUUpdate.resize(nCellsX + 4, std::vector<std::array<double, 4>>(nCellsY + 4));
        std::vector<std::vector<std::array<double, 4>>> uBarDUpdate;
        uBarDUpdate.resize(nCellsX + 4, std::vector<std::array<double, 4>>(nCellsY + 4));

        std::vector<std::vector<std::array<double, 4>>> uTempPlus1;
        uTempPlus1.resize(nCellsX + 4, std::vector<std::array<double, 4>>(nCellsY + 4));
        std::vector<std::vector<std::array<double, 4>>> uPlus1;
        uPlus1.resize(nCellsX + 4, std::vector<std::array<double, 4>>(nCellsY + 4));
        std::vector<std::vector<std::array<double, 4>>> fluxX_SLIC;
        fluxX_SLIC.resize(nCellsX + 3, std::vector<std::array<double, 4>>(nCellsY + 3));
        std::vector<std::vector<std::array<double, 4>>> fluxY_SLIC;
        fluxY_SLIC.resize(nCellsX + 3, std::vector<std::array<double, 4>>(nCellsY + 3));


        //***********************************************************************************************************//
        // compute time step
        startdt = clock();
        double dt = computeTimeStep(u, C, dx, dy, gamma);
        enddt = clock();
        elapsdt += (double)(enddt - startdt) / CLOCKS_PER_SEC;

        t = t + dt;
        std::cout << "ite = " << counter + 1 << ", time = " << t << std::endl;


        //***********************************************************************************************************//
        startx = clock();
        // data reconstruction in x-direction
        for (int i = 1; i < nCellsX + 3; i++) {
            for (int j = 2; j < nCellsY + 2; j++) {
                // reconstruct x-direction
                std::vector<std::array<double, 4>> uBarX_ij = dataReconstruct(u[i - 1][j], u[i][j], u[i + 1][j]);
                uBarL[i][j] = uBarX_ij[0];
                uBarR[i][j] = uBarX_ij[1];
            }
        }

        // half-time-step update in x-direction
        for (int i = 1; i < nCellsX + 3; i++) {
            for (int j = 2; j < nCellsY + 2; j++) {
                std::vector<std::array<double, 4>> uBarUpdateX_ij = halfTimeStepUpdateX(uBarL[i][j], uBarR[i][j], dx, dt, gamma);
                uBarLUpdate[i][j] = uBarUpdateX_ij[0];
                uBarRUpdate[i][j] = uBarUpdateX_ij[1];
            }
        }

        // calculate boundary fluxes in x-direction
        for (int i = 1; i < nCellsX + 2; i++) {
            for (int j = 2; j < nCellsY + 2; j++) {
                fluxX_SLIC[i][j] = getFluxX(uBarRUpdate[i][j], uBarLUpdate[i + 1][j], dx, dt, gamma);
            }
        }

        // add flux in x-direction
        for (int i = 2; i < nCellsX + 2; i++) {
            for (int j = 2; j < nCellsY + 2; j++) {
                for (int k = 0; k < 4; k++) {
                    uTempPlus1[i][j][k] = u[i][j][k] - dt / dx * (fluxX_SLIC[i][j][k] - fluxX_SLIC[i - 1][j][k]);
                }
            }
        }

        endx = clock();
        elapsx += (double)(endx - startx) / CLOCKS_PER_SEC;

        // transmissive boundary condition
        startbc = clock();
        setBoundaryCondition(uTempPlus1, nCellsX, nCellsY);
        endbc = clock();
        elapsbc += (double)(endbc - startbc) / CLOCKS_PER_SEC;

        u = uTempPlus1;  // temporary result


        //***********************************************************************************************************//
        starty = clock();
        // data reconstruction in y-direction
        for (int i = 2; i < nCellsX + 2; i++) {
            for (int j = 1; j < nCellsY + 3; j++) {
                // reconstruct y-direction
                std::vector<std::array<double, 4>> uBarY_ij = dataReconstruct(u[i][j - 1], u[i][j], u[i][j + 1]);
                uBarD[i][j] = uBarY_ij[0];
                uBarU[i][j] = uBarY_ij[1];
            }
        }

        // half-time-step update in y-direction
        for (int i = 2; i < nCellsX + 2; i++) {
            for (int j = 1; j < nCellsY + 3; j++) {
                std::vector<std::array<double, 4>> uBarUpdateY_ij = halfTimeStepUpdateY(uBarD[i][j], uBarU[i][j], dy, dt, gamma);
                uBarDUpdate[i][j] = uBarUpdateY_ij[0];
                uBarUUpdate[i][j] = uBarUpdateY_ij[1];
            }
        }

        // calculate boundary fluxes in y-direction
        for (int i = 2; i < nCellsX + 2; i++) {
            for (int j = 1; j < nCellsY + 2; j++) {
                fluxY_SLIC[i][j] = getFluxY(uBarUUpdate[i][j], uBarDUpdate[i][j + 1], dy, dt, gamma);
            }
        }

        // add flux in y-direction
        for (int i = 2; i < nCellsX + 2; i++) {
            for (int j = 2; j < nCellsY + 2; j++) {
                for (int k = 0; k < 4; k++) {
                    uPlus1[i][j][k] = u[i][j][k] - dt / dy * (fluxY_SLIC[i][j][k] - fluxY_SLIC[i][j - 1][k]);
                }
            }
        }

        endy = clock();
        elapsy += (double)(endy - starty) / CLOCKS_PER_SEC;

        // transmissive boundary condition
        startbc = clock();
        setBoundaryCondition(uPlus1, nCellsX, nCellsY);
        endbc = clock();
        elapsbc += (double)(endbc - startbc) / CLOCKS_PER_SEC;

        u = uPlus1;  // final result for this update-loop
        counter++;

        end = clock();
        elapstotal += (double)(end - start) / CLOCKS_PER_SEC;

        // data recording
        if (Record && t / time_ratio >= record_list[record_index]) {
            std::cout << "Recording: t = " << t << std::endl;
            dataRecord(u, case_id, nCellsX, nCellsY, x0, y0, dx, dy, t / time_ratio, gamma);
            record_index++;
        }

    } while (t < tStop);

    // output time recording
    std::cout << "=== Timing Results ===" << std::endl;
    std::cout << "Total execution time: " << elapstotal << " sec" << std::endl;
    std::cout << "computeTimeStep: " << elapsdt << " sec" << std::endl;
    std::cout << "Boundary Conditions: " << elapsbc << " sec" << std::endl;
    std::cout << "X-direction evolution: " << elapsx << " sec" << std::endl;
    std::cout << "Y-direction evolution: " << elapsy << " sec" << std::endl;

    return 0;
}

