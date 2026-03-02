# This code is written by Jisung Hwang (4011hjs@kaist.ac.kr) in 2024.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.


import torch as th
from .lookup_tables import (
    vitogi,
    lctab_2d,
    pfvtab,
    cpnttab,
    ambtab,
    cstrtab,
    cas_tab,
    ax0_tab,
    ax1_tab,
    axd_tab,
)


class occupancy_dual_contouring:
    def __init__(self, device):
        self.device = device
        self.vitogi = th.tensor(vitogi, dtype=th.int64, device=device)

        for a in lctab_2d:
            a += [[-1, -1, -1]] * (4 - len(a))
        self.lctab_2d = th.tensor(lctab_2d, dtype=th.int64, device=device)

        for a in pfvtab:
            for b in a:
                b += [0] * (15 - len(b))
            a += [[0] * 15] * (4 - len(a))
        self.pfvtab = th.tensor(pfvtab, dtype=th.int64, device=device)

        self.qvntab = th.div(th.sum(self.pfvtab > 0, dim=2), 2, rounding_mode="floor")

        self.vtstab = th.sort(self.pfvtab.reshape(256, -1), dim=1)[0][:, -32:]
        for i in range(4):
            self.vtstab[self.qvntab[:, i] > 0, i] = 25 + i

        self.vbitab = th.zeros((256, 12), dtype=th.int64, device=device)
        for i in range(12):
            tmp = th.sum(self.pfvtab == i + 1, dim=2)
            for j in range(4):
                self.vbitab[tmp[:, j] > 0, i] = j + 9

        self.cstrtab = [[] for i in range(256)]
        for a in cstrtab:
            self.cstrtab[a[0]] = a[1:]
        for a in self.cstrtab:
            a += [[[0, 2, 1]]] * (4 - len(a))
        for a in self.cstrtab:
            for b in a:
                b += [[0, 2, 1]] * (2 - len(b))
        tmp = th.tensor(self.cstrtab, dtype=th.int64, device=device)
        self.cstrtab = th.stack(
            (
                th.div(tmp, 4, rounding_mode="floor"),
                th.div(tmp, 2, rounding_mode="floor") % 2,
                tmp % 2,
            ),
            dim=4,
        ).to(th.float64)

        self.cpnttab = [[] for i in range(256)]
        for a in cpnttab:
            self.cpnttab[a[0]] = a[1:]
        for a in self.cpnttab:
            a += [[12, 12, 12]] * (4 - len(a))
        self.cpnttab = th.tensor(self.cpnttab, dtype=th.float64, device=device) / 24.0

        self.ambtab = [[0, 0, 0, 0] for _ in range(256)]
        for a in ambtab:
            self.ambtab[a[0]] = [1, a[1], a[2], a[3]]
        self.ambtab = th.tensor(self.ambtab, dtype=th.int64, device=device)

        self.axd_tab = th.tensor(axd_tab, dtype=th.int64, device=device)
        self.ax0_tab = th.tensor(ax0_tab, dtype=th.int64, device=device)
        self.ax1_tab = th.tensor(ax1_tab, dtype=th.int64, device=device)
        self.cas_tab = th.tensor(cas_tab, dtype=th.int64, device=device)

    @th.no_grad()
    def extract_mesh(
        self,
        imp_func,
        min_coord: list[float] = [-0.5, -0.5, -0.5],
        max_coord: list[float] = [0.5, 0.5, 0.5],
        num_grid: int = 128,
        isolevel: float = 0.0,
        batch_size: int = 100000000,
        imp_func_cplx: int = 3,
        outside: bool = True,
        BINARY_SEARCH: int = 15,
        VERTICAL_RANGE: float = 0.8,
        VERTICAL_LINEAR_SEARCH: int = 4,
        VERTICAL_BINARY_SEARCH: int = 11,
        HORIZNTL_RANGE: float = 0.71,
        HORIZNTL_LINEAR_SEARCH: int = 3,
        HORIZNTL_BINARY_SEARCH: int = 12,
        ERR: float = 3e-7,
        SEP: float = 1e-3,
        QEF_REG: float = 0.05,
    ):
        GRID = num_grid
        GRID_ = GRID + 1
        GRID2_, GRID3_ = GRID_**2, GRID_**3
        device = self.device

        min_coordc = th.tensor(min_coord, dtype=th.float64, device=device).reshape(1, 3)
        siz_coordc = (
            th.tensor(max_coord, dtype=th.float64, device=device).reshape(1, 3)
            - min_coordc
        )

        B = max(1, batch_size // imp_func_cplx)
        if outside:
            get_occv_ = lambda x: imp_func(x) < isolevel
        else:
            get_occv_ = lambda x: imp_func(x) > isolevel

        pidx1_to_pidx3 = lambda pi1: th.remainder(
            th.stack(
                (
                    th.div(pi1, GRID2_, rounding_mode="floor"),
                    th.div(pi1, GRID_, rounding_mode="floor"),
                    pi1,
                ),
                dim=-1,
            ),
            GRID_,
        )
        pidx3_to_pnts = lambda pi3: (pi3 / GRID).to(th.float64)

        def get_occv(cor_):
            if cor_.shape[0] == 0:
                return th.zeros((0,), dtype=th.bool, device=device)

            cor = min_coordc + cor_ * siz_coordc
            if cor.shape[0] <= B:
                return get_occv_(cor)

            out = []
            for cal_i in range(0, cor.shape[0], B):
                out.append(get_occv_(cor[cal_i : cal_i + B]))
            return th.cat(out, dim=0)

        pidx1 = th.arange(GRID3_, dtype=th.int64, device=device)
        occs = get_occv(pidx3_to_pnts(pidx1_to_pidx3(pidx1))).reshape(
            GRID_, GRID_, GRID_
        )

        vals = th.zeros((GRID, GRID, GRID), dtype=th.uint8, device=device)
        vals += occs[1:, 1:, 1:]
        vals <<= 1
        vals += occs[1:, 1:, :-1]
        vals <<= 1
        vals += occs[1:, :-1, 1:]
        vals <<= 1
        vals += occs[1:, :-1, :-1]
        vals <<= 1
        vals += occs[:-1, 1:, 1:]
        vals <<= 1
        vals += occs[:-1, 1:, :-1]
        vals <<= 1
        vals += occs[:-1, :-1, 1:]
        vals <<= 1
        vals += occs[:-1, :-1, :-1]

        cel_pidx3 = th.argwhere((vals > 0) & (vals < 255))

        val_ = vals[cel_pidx3[:, 0], cel_pidx3[:, 1], cel_pidx3[:, 2]].to(th.int64)
        ambres = self.ambtab[val_]
        amb_idcs = th.argwhere(ambres[:, 0] == 1).reshape(-1)
        amb_pidx3 = cel_pidx3[amb_idcs] + ambres[amb_idcs, 1:]
        tpi = th.argwhere(
            th.sum((0 <= amb_pidx3) & (amb_pidx3 < GRID), dim=1) == 3
        ).reshape(-1)
        tmp = vals[amb_pidx3[tpi, 0], amb_pidx3[tpi, 1], amb_pidx3[tpi, 2]].to(th.int64)
        amb_pidx3 = amb_pidx3[tpi[self.ambtab[tmp, 0] == 1]]
        tmp = vals[amb_pidx3[:, 0], amb_pidx3[:, 1], amb_pidx3[:, 2]]
        vals[amb_pidx3[:, 0], amb_pidx3[:, 1], amb_pidx3[:, 2]] = 255 - tmp
        amb_chks = th.zeros((GRID_, GRID_, GRID_), dtype=th.bool, device=device)
        amb_chks[amb_pidx3[:, 0], amb_pidx3[:, 1], amb_pidx3[:, 2]] = True

        val_ = vals[cel_pidx3[:, 0], cel_pidx3[:, 1], cel_pidx3[:, 2]].to(th.int64)

        if cel_pidx3.shape[0] == 0:
            print("Occupancy Dual Contouring: No Surface Detected.")
            return th.zeros((0, 3), dtype=th.float64, device=device), th.zeros(
                (0, 3), dtype=th.int64, device=device
            )

        vtcs = self.vtstab[val_]
        sel_pidx1 = (
            self.vitogi[vtcs, 3] * GRID3_
            + (cel_pidx3[:, 0:1] + self.vitogi[vtcs, 0]) * GRID2_
            + (cel_pidx3[:, 1:2] + self.vitogi[vtcs, 1]) * GRID_
            + (cel_pidx3[:, 2:3] + self.vitogi[vtcs, 2])
        )
        sel_pidx1 = sel_pidx1.reshape(-1)
        sel_pidx1 = th.unique(sel_pidx1[sel_pidx1 >= 0])

        val_pnts = th.zeros((sel_pidx1.shape[0], 3), dtype=th.float64, device=device)

        def get_idx(q):
            S = q.shape
            q_ = q.reshape(-1)
            std, idx = th.unique(th.cat((q_, sel_pidx1), dim=0), return_inverse=True)
            return idx[: q_.shape[0]].reshape(S)

        def _1d_search(pidx3, axis, val_):
            P = pidx3.shape[0]
            pnts = pidx3_to_pnts(pidx3)
            left = th.zeros((P, 1), dtype=th.float64, device=device)
            rght = th.ones((P, 1), dtype=th.float64, device=device)

            for _ in range(BINARY_SEARCH):
                mid = (left + rght) * 0.5
                occv = get_occv(pnts + axis * (mid / GRID))

                jdge = (occv == val_).unsqueeze(1)
                left = th.where(jdge, mid, left)
                rght = th.where(jdge, rght, mid)

            return left * axis

        start_idx = 0
        pidx1 = sel_pidx1[(0 * GRID3_ <= sel_pidx1) & (sel_pidx1 < 3 * GRID3_)]
        count_idx = pidx1.shape[0]

        axis = th.zeros((count_idx, 3), dtype=th.float64, device=device)
        axis[(0 * GRID3_ <= pidx1) & (pidx1 < 1 * GRID3_), 2] = 1.0
        axis[(1 * GRID3_ <= pidx1) & (pidx1 < 2 * GRID3_), 1] = 1.0
        axis[(2 * GRID3_ <= pidx1) & (pidx1 < 3 * GRID3_), 0] = 1.0

        pidx3 = pidx1_to_pidx3(pidx1)
        val_ = occs[pidx3[:, 0], pidx3[:, 1], pidx3[:, 2]]

        val_pnts[start_idx : start_idx + count_idx] = _1d_search(pidx3, axis, val_)
        start_idx += count_idx

        def _2d_linbin(pnts, spnt, locc, axis, srch_range, lin_step, bin_step):
            P = pnts.shape[0]
            tstep = srch_range / lin_step
            lidx = th.full((P,), lin_step, dtype=th.int64, device=device)
            locc = locc.reshape(-1)

            for lin_i in range(lin_step):
                cor_ = pnts + (spnt + axis * ((lin_step - lin_i) * tstep)) / GRID
                occv = get_occv(cor_)
                lidx = th.where(occv != locc, lin_step - lin_i - 1, lidx)

            left = spnt + axis * (lidx.unsqueeze(1) * tstep)
            rght = left + axis * tstep
            locc = locc.unsqueeze(1)

            for bin_i in range(bin_step):
                mid = (left + rght) * 0.5
                occv = get_occv(pnts + mid / GRID).unsqueeze(1)

                jdge = locc == occv
                left = th.where(jdge, mid, left)
                rght = th.where(jdge, rght, mid)

            return left

        def _2d_search(pidx3, axdir, axis0, axis1, val_, cases):
            P = pidx3.shape[0]
            arng = th.arange(P, dtype=th.int64, device=device)
            pnts = pidx3_to_pnts(pidx3)
            axis0, axis1 = axis0.to(th.float64), axis1.to(th.float64)

            cidx3 = pidx3.clone()
            lcres = self.lctab_2d[val_, cases]
            cidx3[arng, axdir[:, 0]] += lcres[:, 0]
            cidx3[arng, axdir[:, 1]] += lcres[:, 1]
            edir = axdir[arng, lcres[:, 2]]
            cidx1 = (
                (2 - edir) * GRID3_
                + cidx3[:, 0] * GRID2_
                + cidx3[:, 1] * GRID_
                + cidx3[:, 2]
            )
            linv = val_pnts[get_idx(cidx1)]
            lcord = th.where(lcres[:, 2:] == 0, linv, axis0 * lcres[:, 0:1]) + th.where(
                lcres[:, 2:] == 1, linv, axis1 * lcres[:, 1:2]
            )

            chk0 = cidx3.clone()
            occ0 = occs[chk0[:, 0], chk0[:, 1], chk0[:, 2]].unsqueeze(1)
            chk0 = (chk0 - pidx3).to(th.float64)

            chk1 = cidx3.clone()
            chk1[arng, edir] += 1
            occ1 = occs[chk1[:, 0], chk1[:, 1], chk1[:, 2]].unsqueeze(1)
            chk1 = (chk1 - pidx3).to(th.float64)

            cidx3 = pidx3.clone()
            lcres = self.lctab_2d[val_, cases + 1]
            cidx3[arng, axdir[:, 0]] += lcres[:, 0]
            cidx3[arng, axdir[:, 1]] += lcres[:, 1]
            edir = axdir[arng, lcres[:, 2]]
            cidx1 = (
                (2 - edir) * GRID3_
                + cidx3[:, 0] * GRID2_
                + cidx3[:, 1] * GRID_
                + cidx3[:, 2]
            )
            linv = val_pnts[get_idx(cidx1)]
            rcord = th.where(lcres[:, 2:] == 0, linv, axis0 * lcres[:, 0:1]) + th.where(
                lcres[:, 2:] == 1, linv, axis1 * lcres[:, 1:2]
            )

            chk2 = cidx3.clone()
            occ2 = occs[chk2[:, 0], chk2[:, 1], chk2[:, 2]].unsqueeze(1)
            chk2 = (chk2 - pidx3).to(th.float64)

            chk3 = cidx3.clone()
            chk3[arng, edir] += 1
            occ3 = occs[chk3[:, 0], chk3[:, 1], chk3[:, 2]].unsqueeze(1)
            chk3 = (chk3 - pidx3).to(th.float64)

            spnt = (lcord + rcord) * 0.5
            locc = get_occv(pnts + spnt / GRID).unsqueeze(1)

            hnorm = lcord - rcord
            leng = th.norm(hnorm, dim=1, keepdim=True)
            cndt = leng < ERR
            leng[cndt] = 1.0
            hnorm = hnorm / leng
            vnorm = th.cross(th.cross(axis0, axis1, dim=1), hnorm, dim=1)

            tmp = th.sum((chk0 - spnt) * vnorm, dim=1, keepdim=True)
            vnorm = th.where(
                (th.abs(tmp) > ERR) & th.bitwise_xor(tmp > 0, locc != occ0),
                -vnorm,
                vnorm,
            )

            tmp = th.sum((chk1 - spnt) * vnorm, dim=1, keepdim=True)
            vnorm = th.where(
                (th.abs(tmp) > ERR) & th.bitwise_xor(tmp > 0, locc != occ1),
                -vnorm,
                vnorm,
            )

            tmp = th.sum((chk2 - spnt) * vnorm, dim=1, keepdim=True)
            vnorm = th.where(
                (th.abs(tmp) > ERR) & th.bitwise_xor(tmp > 0, locc != occ2),
                -vnorm,
                vnorm,
            )

            tmp = th.sum((chk3 - spnt) * vnorm, dim=1, keepdim=True)
            vnorm = th.where(
                (th.abs(tmp) > ERR) & th.bitwise_xor(tmp > 0, locc != occ3),
                -vnorm,
                vnorm,
            )

            left = _2d_linbin(
                pnts,
                spnt,
                locc,
                vnorm,
                VERTICAL_RANGE,
                VERTICAL_LINEAR_SEARCH,
                VERTICAL_BINARY_SEARCH,
            )
            vl = th.sum((left - spnt) * vnorm, dim=1, keepdim=True)
            cndt |= th.abs(vl) < ERR

            lleft = (
                _2d_linbin(
                    pnts,
                    left,
                    locc,
                    hnorm,
                    HORIZNTL_RANGE,
                    HORIZNTL_LINEAR_SEARCH,
                    HORIZNTL_BINARY_SEARCH,
                )
                - lcord
            )
            lh = th.sum(lleft * hnorm, dim=1, keepdim=True)
            rleft = (
                _2d_linbin(
                    pnts,
                    left,
                    locc,
                    -hnorm,
                    HORIZNTL_RANGE,
                    HORIZNTL_LINEAR_SEARCH,
                    HORIZNTL_BINARY_SEARCH,
                )
                - rcord
            )
            rh = th.sum(rleft * hnorm, dim=1, keepdim=True)

            denm = rh - lh
            cndt |= th.abs(denm) < ERR
            rslt = ((0.5 * (rh + lh)) * hnorm + vl * vnorm) * (leng / denm)

            cndt |= th.sum(rslt * vnorm, dim=1, keepdim=True) < 0
            rslt = th.where(cndt, th.zeros_like(rslt), rslt)
            return spnt + rslt

        pidx1 = sel_pidx1[(3 * GRID3_ <= sel_pidx1) & (sel_pidx1 < 9 * GRID3_)]
        count_idx = pidx1.shape[0]

        csidx = th.div(pidx1, GRID3_, rounding_mode="floor") - 3
        axdir = self.axd_tab[csidx]
        axis0 = self.ax0_tab[csidx]
        axis1 = self.ax1_tab[csidx]
        cases = self.cas_tab[csidx]
        axis2 = axis0 + axis1

        pidx3 = pidx1_to_pidx3(pidx1)
        val_ = (
            occs[pidx3[:, 0], pidx3[:, 1], pidx3[:, 2]]
            + 2
            * occs[
                pidx3[:, 0] + axis1[:, 0],
                pidx3[:, 1] + axis1[:, 1],
                pidx3[:, 2] + axis1[:, 2],
            ]
            + 4
            * occs[
                pidx3[:, 0] + axis0[:, 0],
                pidx3[:, 1] + axis0[:, 1],
                pidx3[:, 2] + axis0[:, 2],
            ]
            + 8
            * occs[
                pidx3[:, 0] + axis2[:, 0],
                pidx3[:, 1] + axis2[:, 1],
                pidx3[:, 2] + axis2[:, 2],
            ]
        )
        tmp = amb_chks[pidx3[:, 0], pidx3[:, 1], pidx3[:, 2]]
        val_[tmp] = 15 - val_[tmp]

        val_pnts[start_idx : start_idx + count_idx] = _2d_search(
            pidx3, axdir, axis0, axis1, val_, cases
        )
        start_idx += count_idx

        def _solve_QEF(pidx3, val_, cases):
            out = th.zeros((pidx3.shape[0], 3), dtype=th.float64, device=device)

            for i in range(3, 8):
                idx = th.argwhere(self.qvntab[val_, cases] == i).reshape(-1)
                pid3_ = pidx3[idx]
                val0_ = val_[idx]
                cas = cases[idx]
                vtcs = self.pfvtab[val0_, cas, : 2 * i + 1]
                vitg = self.vitogi[vtcs]

                pnti = (
                    vitg[:, :, 3] * GRID3_
                    + (pid3_[:, 0:1] + vitg[:, :, 0]) * GRID2_
                    + (pid3_[:, 1:2] + vitg[:, :, 1]) * GRID_
                    + (pid3_[:, 2:3] + vitg[:, :, 2])
                )

                pidx = get_idx(pnti)
                vert = val_pnts[pidx]
                vert += vitg[:, :, :3]

                _1vt = vert[:, 1::2]
                qnrm = th.cross(_1vt - vert[:, :-1:2], _1vt - vert[:, 2::2], dim=2)
                cntr = th.mean(_1vt, dim=1)

                anrm = th.norm(qnrm, dim=2, keepdim=True)
                cndt = anrm < ERR
                qnrm /= th.where(cndt, th.ones_like(anrm), anrm)
                cndt = th.sum(cndt, dim=1) == cndt.shape[1]

                eye_ = (
                    th.eye(3, dtype=th.float64, device=device)
                    .view(1, 3, 3)
                    .repeat(cntr.shape[0], 1, 1)
                )
                qefr = th.linalg.lstsq(
                    th.cat((qnrm, QEF_REG * eye_), dim=1),
                    th.cat(
                        (
                            th.sum(_1vt * qnrm, dim=2, keepdim=True),
                            QEF_REG * cntr.unsqueeze(2),
                        ),
                        dim=1,
                    ),
                ).solution.reshape(-1, 3)
                qefr = th.where(cndt, cntr, qefr)

                cpnt = self.cpnttab[val0_, cas]
                ctoq = qefr - cpnt
                leng = th.norm(ctoq, dim=1, keepdim=True)

                cscd = self.cstrtab[val0_, cas]
                cvtx = cscd[:, :, 0]
                cnml = th.cross(cscd[:, :, 2] - cvtx, cscd[:, :, 1] - cvtx, dim=2)
                t = th.sum(cnml * (cvtx - cpnt.unsqueeze(1)), dim=2) / th.sum(
                    cnml * ctoq.unsqueeze(1), dim=2
                )

                t = th.cat(((1.0 - cpnt) / ctoq, -cpnt / ctoq, t), dim=1) * leng
                cndt = (t < 0) | th.isnan(t)
                t[cndt] = leng.repeat(1, 8)[cndt] + SEP
                t = (t - SEP) / th.where(leng < ERR, th.ones_like(leng), leng)
                t = th.min(th.clamp(t, 0.0, 1.0), dim=1, keepdim=True)[0]

                out[idx] = cpnt + t * ctoq

            return out

        pidx1 = sel_pidx1[9 * GRID3_ <= sel_pidx1]
        count_idx = pidx1.shape[0]
        pidx3 = pidx1_to_pidx3(pidx1)
        val_ = vals[pidx3[:, 0], pidx3[:, 1], pidx3[:, 2]].to(th.int64)
        cases = th.div(pidx1, GRID3_, rounding_mode="floor") - 9

        val_pnts[start_idx : start_idx + count_idx] = _solve_QEF(pidx3, val_, cases)
        start_idx += count_idx

        def _get_face():
            faces = []

            cndts, vidcs, vpnts = [], [], []
            evofs = []
            idcs = th.tensor(
                [[[12], [8], [6], [1]], [[11], [4], [9], [2]], [[10], [7], [5], [3]]],
                dtype=th.int64,
                device=device,
            )
            ofst = th.tensor(
                [
                    [[0], [GRID_], [GRID2_], [GRID2_ + GRID_]],
                    [[0], [GRID2_], [1], [GRID2_ + 1]],
                    [[0], [1], [GRID_], [1 + GRID_]],
                ],
                dtype=th.int64,
                device=device,
            )
            vofs = th.tensor(
                [
                    [[0, 0, 0], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 1, 0]],
                    [[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1], [1, 0, 1]],
                    [[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [0, 1, 1]],
                ],
                dtype=th.int64,
                device=device,
            ).unsqueeze(2)

            z, y, x = th.argwhere(
                th.bitwise_xor(occs[1:-1, 1:-1, :-1], occs[1:-1, 1:-1, 1:])
            ).T
            val_ = vals[
                z.unsqueeze(0) + vofs[0, :4, 0, 0:1],
                y.unsqueeze(0) + vofs[0, :4, 0, 1:2],
                x.unsqueeze(0) + vofs[0, :4, 0, 2:3],
            ].to(th.int64)
            fac_idx = (z * GRID2_ + y * GRID_ + x).unsqueeze(0)
            vidx3 = self.vbitab[val_, idcs[0] - 1] * GRID3_ + fac_idx + ofst[0]
            gvidx = get_idx(th.cat((vidx3, fac_idx + (GRID2_ + GRID_)), dim=0))
            gvpnt = val_pnts[gvidx] + vofs[0].to(th.float64)

            cndts.append(occs[z + 1, y + 1, x + 1])
            vidcs.append(gvidx.clone())
            vpnts.append(gvpnt.clone())
            evofs.append(vofs[0, 4].repeat(x.shape[0], 1))

            z, y, x = th.argwhere(
                th.bitwise_xor(occs[1:-1, :-1, 1:-1], occs[1:-1, 1:, 1:-1])
            ).T
            val_ = vals[
                z.unsqueeze(0) + vofs[1, :4, 0, 0:1],
                y.unsqueeze(0) + vofs[1, :4, 0, 1:2],
                x.unsqueeze(0) + vofs[1, :4, 0, 2:3],
            ].to(th.int64)
            fac_idx = (z * GRID2_ + y * GRID_ + x).unsqueeze(0)
            vidx3 = self.vbitab[val_, idcs[1] - 1] * GRID3_ + fac_idx + ofst[1]
            gvidx = get_idx(th.cat((vidx3, fac_idx + (GRID3_ + GRID2_ + 1)), dim=0))
            gvpnt = val_pnts[gvidx] + vofs[1].to(th.float64)

            cndts.append(occs[z + 1, y + 1, x + 1])
            vidcs.append(gvidx.clone())
            vpnts.append(gvpnt.clone())
            evofs.append(vofs[1, 4].repeat(x.shape[0], 1))

            z, y, x = th.argwhere(
                th.bitwise_xor(occs[:-1, 1:-1, 1:-1], occs[1:, 1:-1, 1:-1])
            ).T
            val_ = vals[
                z.unsqueeze(0) + vofs[2, :4, 0, 0:1],
                y.unsqueeze(0) + vofs[2, :4, 0, 1:2],
                x.unsqueeze(0) + vofs[2, :4, 0, 2:3],
            ].to(th.int64)
            fac_idx = (z * GRID2_ + y * GRID_ + x).unsqueeze(0)
            vidx3 = self.vbitab[val_, idcs[2] - 1] * GRID3_ + fac_idx + ofst[2]
            gvidx = get_idx(th.cat((vidx3, fac_idx + (2 * GRID3_ + 1 + GRID_)), dim=0))
            gvpnt = val_pnts[gvidx] + vofs[2].to(th.float64)

            cndts.append(occs[z + 1, y + 1, x + 1])
            vidcs.append(gvidx.clone())
            vpnts.append(gvpnt.clone())
            evofs.append(vofs[2, 4].repeat(x.shape[0], 1))

            cndts, vidcs, vpnts = (
                th.cat(cndts, dim=0),
                th.cat(vidcs, dim=1),
                th.cat(vpnts, dim=1),
            )
            evofs = th.cat(evofs, dim=0).to(th.float64).unsqueeze(0)

            idcs = th.tensor([0, 1, 3, 2, 0, 1], dtype=th.int64, device=device)
            pnts1 = vpnts[idcs[0:4]]
            pnts2 = vpnts[idcs[1:5]]
            pnts3 = vpnts[idcs[2:6]]
            sitst = (
                th.sum(
                    th.cross(pnts1 - evofs, pnts3 - evofs, dim=2) * (pnts2 - evofs),
                    dim=2,
                )
                < 0
            )
            sitst |= (
                th.sum(th.cross(pnts1 - 1.0, pnts3 - 1.0, dim=2) * (pnts2 - 1.0), dim=2)
                > 0
            )

            sit12 = sitst[0] | sitst[2]
            sit03 = sitst[1] | sitst[3]
            sitsp = sit12 & sit03
            sitnp = ~sitsp

            idcs0, idcs1, idcs2, idcs3, eidcs = vidcs[:, sitsp]
            faces = th.cat(
                (
                    th.stack((idcs1, idcs0, eidcs), dim=1),
                    th.stack((idcs3, idcs1, eidcs), dim=1),
                    th.stack((idcs2, idcs3, eidcs), dim=1),
                    th.stack((idcs0, idcs2, eidcs), dim=1),
                ),
                dim=0,
            )

            idcs0, idcs1, idcs2, idcs3, _ = vidcs[:, sitnp]
            pnts0, pnts1, pnts2, pnts3, epnts = vpnts[:, sitnp]
            sit12, sit03 = sit12[sitnp], sit03[sitnp]

            v03 = pnts3 - pnts0
            leng = th.norm(v03, dim=1)
            cnn03 = leng < ERR
            v03 /= th.where(cnn03, th.ones_like(leng), leng).unsqueeze(1)

            ev0s = epnts - pnts0
            t_ = th.sum(ev0s * v03, dim=1)
            vp03 = pnts0 + t_.unsqueeze(1) * v03

            n1 = th.cross(v03, pnts1 - pnts0, dim=1)
            ln1 = th.norm(n1, dim=1)
            bnn03 = ln1 < ERR
            t_ = th.sum(ev0s * n1, dim=1) / th.clamp(ln1, 1e-5) ** 2
            vp1 = epnts - t_.unsqueeze(1) * n1
            cndt = th.sum(th.cross(v03, vp1 - pnts0, dim=1) * n1, dim=1) > 0
            vp1 = th.where(cndt.unsqueeze(1), vp1, vp03)

            n2 = th.cross(v03, pnts2 - pnts0, dim=1)
            ln2 = th.norm(n2, dim=1)
            bnn03 |= ln2 < ERR
            t_ = th.sum(ev0s * n2, dim=1) / th.clamp(ln2, 1e-5) ** 2
            vp2 = epnts - t_.unsqueeze(1) * n2
            cndt = th.sum(th.cross(v03, vp2 - pnts0, dim=1) * n2, dim=1) > 0
            vp2 = th.where(cndt.unsqueeze(1), vp2, vp03)

            lf03 = th.minimum(th.norm(vp1 - epnts, dim=1), th.norm(vp2 - epnts, dim=1))

            v12 = pnts2 - pnts1
            leng = th.norm(v12, dim=1)
            cnn12 = leng < ERR
            v12 /= th.where(cnn12, th.ones_like(leng), leng).unsqueeze(1)

            ev1s = epnts - pnts1
            t_ = th.sum(ev1s * v12, dim=1)
            vp12 = pnts1 + t_.unsqueeze(1) * v12

            n0 = th.cross(v12, pnts0 - pnts1, dim=1)
            ln0 = th.norm(n0, dim=1)
            t_ = th.sum(ev1s * n0, dim=1) / th.clamp(ln0, 1e-5) ** 2
            vp0 = epnts - t_.unsqueeze(1) * n0
            cndt = th.sum(th.cross(v12, vp0 - pnts1, dim=1) * n0, dim=1) > 0
            vp0 = th.where(cndt.unsqueeze(1), vp0, vp12)

            n3 = th.cross(v12, pnts3 - pnts1, dim=1)
            ln3 = th.norm(n3, dim=1)
            t_ = th.sum(ev1s * n3, dim=1) / th.clamp(ln3, 1e-5) ** 2
            vp3 = epnts - t_.unsqueeze(1) * n3
            cndt = th.sum(th.cross(v12, vp3 - pnts1, dim=1) * n3, dim=1) > 0
            vp3 = th.where(cndt.unsqueeze(1), vp3, vp12)

            lf12 = th.minimum(th.norm(vp0 - epnts, dim=1), th.norm(vp3 - epnts, dim=1))

            cnn12 |= (lf12 < lf03) | bnn03 | sit12
            cnn12 &= ~(cnn03 | sit03)

            faces = th.cat(
                (
                    th.where(
                        cnn12.unsqueeze(1),
                        th.stack((idcs0, idcs2, idcs1), dim=1),
                        th.stack((idcs0, idcs3, idcs1), dim=1),
                    ),
                    th.where(
                        cnn12.unsqueeze(1),
                        th.stack((idcs1, idcs2, idcs3), dim=1),
                        th.stack((idcs0, idcs2, idcs3), dim=1),
                    ),
                    faces,
                ),
                dim=0,
            )

            cndtn, cndts = cndts[sitnp], cndts[sitsp]
            cndts = th.cat((cndtn, cndtn, cndts, cndts, cndts, cndts))
            ftmp = faces[cndts, 1:3].clone()
            faces[cndts, 1] = ftmp[:, 1]
            faces[cndts, 2] = ftmp[:, 0]

            return th.unique(faces, return_inverse=True)

        vidcs, faces = _get_face()

        pidx1 = sel_pidx1[vidcs]
        pnts = pidx3_to_pnts(pidx1_to_pidx3(pidx1)) + val_pnts[vidcs] / GRID
        verts = min_coordc + pnts * siz_coordc

        return verts, faces
