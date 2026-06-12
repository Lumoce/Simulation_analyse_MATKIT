"""
电荷密度分析模块 / Charge Density Analysis Module
===================================================

提供差分电荷密度计算、平面平均、宏观平均、电荷积分、
电荷转移分析和Gaussian cube格式导出等功能。
Provides differential charge density calculation, planar averaging,
macroscopic averaging, charge integration, charge transfer analysis,
and Gaussian cube file export.

依赖 / Dependencies:
    - numpy

使用示例 / Usage Example:
    >>> from matkit.charge import calc_diff_charge_density, planar_average
    >>> diff_rho = calc_diff_charge_density(chgcar_total, chgcar_slab, chgcar_ads)
    >>> avg = planar_average(diff_rho, axis=2)
"""

import numpy as np
import warnings


def calc_diff_charge_density(chgcar_total, chgcar_slab, chgcar_adsorbate):
    """
    计算差分电荷密度 / Calculate differential charge density.

    通过减去衬底和吸附物的电荷密度来获得差分电荷密度，
    用于分析吸附过程中的电荷重新分布。
    Computes the differential charge density by subtracting the charge
    densities of the slab and adsorbate from the total system, useful
    for analyzing charge redistribution during adsorption.

    公式 / Formula:
        Δρ = ρ_total - ρ_slab - ρ_adsorbate

    参数 / Parameters:
        chgcar_total (dict): 总系统的CHGCAR数据字典，需包含 'total_charge' 键（3D numpy数组）。
            CHGCAR data dict of the total system. Must contain 'total_charge' key (3D numpy array).
        chgcar_slab (dict): 衬底的CHGCAR数据字典，需包含 'total_charge' 键（3D numpy数组）。
            CHGCAR data dict of the clean slab. Must contain 'total_charge' key (3D numpy array).
        chgcar_adsorbate (dict): 吸附物的CHGCAR数据字典，需包含 'total_charge' 键（3D numpy数组）。
            CHGCAR data dict of the isolated adsorbate. Must contain 'total_charge' key (3D numpy array).

    返回 / Returns:
        numpy.ndarray: 差分电荷密度3D数组，形状与输入相同。
            3D numpy array of differential charge density, same shape as inputs.

    异常 / Raises:
        ValueError: 如果输入数据形状不匹配或缺少必需的键。
            If input data shapes do not match or required keys are missing.
        TypeError: 如果输入不是字典或total_charge不是numpy数组。
            If inputs are not dicts or total_charge is not a numpy array.

    示例 / Example:
        >>> diff_rho = calc_diff_charge_density(total_chg, slab_chg, ads_chg)
        >>> print(f"Max diff charge density: {diff_rho.max():.4f} e/Å³")
    """
    # 输入验证 / Input validation
    for name, data in [("chgcar_total", chgcar_total),
                       ("chgcar_slab", chgcar_slab),
                       ("chgcar_adsorbate", chgcar_adsorbate)]:
        if not isinstance(data, dict):
            raise TypeError(
                f"参数 '{name}' 必须是字典类型 / "
                f"Parameter '{name}' must be a dict, got {type(data).__name__}"
            )
        if "total_charge" not in data:
            raise ValueError(
                f"参数 '{name}' 缺少 'total_charge' 键 / "
                f"Parameter '{name}' is missing the 'total_charge' key"
            )
        if not isinstance(data["total_charge"], np.ndarray):
            raise TypeError(
                f"参数 '{name}['total_charge']' 必须是numpy数组 / "
                f"Parameter '{name}['total_charge']' must be a numpy array, "
                f"got {type(data['total_charge']).__name__}"
            )

    rho_total = chgcar_total["total_charge"]
    rho_slab = chgcar_slab["total_charge"]
    rho_ads = chgcar_adsorbate["total_charge"]

    # 检查网格形状一致性 / Check grid shape consistency
    if rho_total.shape != rho_slab.shape:
        raise ValueError(
            f"总系统与衬底的电荷密度网格形状不匹配: "
            f"total={rho_total.shape}, slab={rho_slab.shape}。"
            f"请确保使用相同的网格尺寸和FFT参数。 / "
            f"Charge density grid shapes do not match between total and slab: "
            f"total={rho_total.shape}, slab={rho_slab_shape}. "
            f"Ensure same grid size and FFT settings."
        )

    if rho_total.shape != rho_ads.shape:
        raise ValueError(
            f"总系统与吸附物的电荷密度网格形状不匹配: "
            f"total={rho_total.shape}, adsorbate={rho_ads.shape}。"
            f"请确保使用相同的网格尺寸和FFT参数。 / "
            f"Charge density grid shapes do not match between total and adsorbate: "
            f"total={rho_total.shape}, adsorbate={rho_ads.shape}. "
            f"Ensure same grid size and FFT settings."
        )

    diff_density = rho_total - rho_slab - rho_ads

    return diff_density


def planar_average(charge_density, axis=2):
    """
    计算电荷密度的平面平均 / Calculate planar average of charge density.

    沿指定轴方向对电荷密度进行平均，得到一维的平面平均电荷密度分布。
    常用于分析表面偶极矩和功函数变化。
    Averages the charge density along the specified axis direction,
    yielding a 1D planar-averaged charge density profile.
    Commonly used for analyzing surface dipole moments and work function changes.

    参数 / Parameters:
        charge_density (numpy.ndarray): 3D电荷密度数组。
            3D charge density array.
        axis (int, optional): 沿哪个轴进行平面平均。默认为2（z方向）。
            Axis along which to compute the planar average.
            Default is 2 (z-direction).

    返回 / Returns:
        numpy.ndarray: 沿指定轴的1D平面平均电荷密度数组。
            1D planar-averaged charge density array along the specified axis.

    异常 / Raises:
        ValueError: 如果charge_density不是3D数组或axis无效。
            If charge_density is not 3D or axis is invalid.

    示例 / Example:
        >>> avg = planar_average(diff_rho, axis=2)
        >>> print(f"Planar average shape: {avg.shape}")
    """
    if not isinstance(charge_density, np.ndarray):
        raise TypeError(
            f"charge_density 必须是numpy数组 / "
            f"charge_density must be a numpy array, got {type(charge_density).__name__}"
        )

    if charge_density.ndim != 3:
        raise ValueError(
            f"charge_density 必须是3D数组，当前维度为 {charge_density.ndim} / "
            f"charge_density must be a 3D array, got {charge_density.ndim}D"
        )

    if axis not in (0, 1, 2):
        raise ValueError(
            f"axis 必须是 0、1 或 2，当前值为 {axis} / "
            f"axis must be 0, 1, or 2, got {axis}"
        )

    planar_avg = np.mean(charge_density, axis=axis)

    return planar_avg


def macroscopic_average(planar_avg, period):
    """
    计算宏观平均电荷密度 / Calculate macroscopic average of charge density.

    使用余弦傅里叶滤波器去除原子尺度的振荡，得到宏观平均电荷密度。
    这对于确定真空层中的电势和计算表面偶极矩非常重要。
    Applies a cosine Fourier filter to remove atomic-scale oscillations,
    yielding the macroscopic average charge density. This is essential
    for determining the potential in the vacuum region and calculating
    surface dipole moments.

    方法 / Method:
        使用截断频率为 2π/period 的余弦窗口函数对平面平均数据进行
        傅里叶滤波。周期参数应略大于层间原子间距。
        Uses a cosine window function with a cutoff frequency of 2π/period
        to Fourier-filter the planar-averaged data. The period parameter
        should be slightly larger than the interlayer atomic spacing.

    参数 / Parameters:
        planar_avg (numpy.ndarray): 1D平面平均电荷密度数组。
            1D planar-averaged charge density array.
        period (float): 滤波周期（以网格点数为单位），应略大于原子层间距。
            Filter period in units of grid points. Should be slightly larger
            than the interlayer atomic spacing.

    返回 / Returns:
        numpy.ndarray: 1D宏观平均电荷密度数组，与输入形状相同。
            1D macroscopic-averaged charge density array, same shape as input.

    异常 / Raises:
        ValueError: 如果planar_avg不是1D数组或period无效。
            If planar_avg is not 1D or period is invalid.

    参考 / References:
        Zhang, Y. & Yang, W. (1999). Phys. Rev. B, 59, 8231.
        Neugebauer, J. & Scheffler, M. (1992). Phys. Rev. B, 46, 16067.

    示例 / Example:
        >>> macro_avg = macroscopic_average(planar_avg, period=12)
    """
    if not isinstance(planar_avg, np.ndarray):
        raise TypeError(
            f"planar_avg 必须是numpy数组 / "
            f"planar_avg must be a numpy array, got {type(planar_avg).__name__}"
        )

    if planar_avg.ndim != 1:
        raise ValueError(
            f"planar_avg 必须是1D数组，当前维度为 {planar_avg.ndim} / "
            f"planar_avg must be a 1D array, got {planar_avg.ndim}D"
        )

    if not isinstance(period, (int, float)) or period <= 0:
        raise ValueError(
            f"period 必须为正数，当前值为 {period} / "
            f"period must be a positive number, got {period}"
        )

    N = len(planar_avg)

    # 对平面平均数据进行傅里叶变换 / Fourier transform the planar average
    fourier = np.fft.fft(planar_avg)

    # 构建余弦窗口函数 / Build cosine window function
    G = np.fft.fftfreq(N, d=1.0 / N)  # 频率网格 / Frequency grid
    G_cutoff = N / period  # 截断频率 / Cutoff frequency

    # 余弦窗口: 在0到G_cutoff之间为1，在G_cutoff到2*G_cutoff之间余弦衰减
    # Cosine window: 1 from 0 to G_cutoff, cosine decay from G_cutoff to 2*G_cutoff
    window = np.zeros(N)
    for i in range(N):
        g_abs = abs(G[i])
        if g_abs <= G_cutoff:
            window[i] = 1.0
        elif g_abs <= 2.0 * G_cutoff:
            window[i] = 0.5 * (1.0 + np.cos(np.pi * (g_abs - G_cutoff) / G_cutoff))

    # 应用滤波器并逆变换 / Apply filter and inverse transform
    filtered_fourier = fourier * window
    macro_avg = np.real(np.fft.ifft(filtered_fourier))

    return macro_avg


def integrate_charge_in_region(charge_density, lattice, z_min, z_max):
    """
    在指定z范围内积分电荷密度 / Integrate charge density within a specified z-range.

    计算给定z范围内的总电荷量，可用于分析表面偶极矩、
    吸附物电荷转移量等。
    Computes the total charge within a given z-range, useful for analyzing
    surface dipole moments, adsorbate charge transfer, etc.

    参数 / Parameters:
        charge_density (numpy.ndarray): 3D电荷密度数组（单位: e/Å³）。
            3D charge density array (units: e/Å³).
        lattice (numpy.ndarray): 3x3晶格向量矩阵（单位: Å），每行为一个晶格矢量。
            3x3 lattice vector matrix (units: Å), each row is a lattice vector.
        z_min (float): z方向积分下限（单位: Å）。
            Lower z-boundary for integration (units: Å).
        z_max (float): z方向积分上限（单位: Å）。
            Upper z-boundary for integration (units: Å).

    返回 / Returns:
        float: 指定z范围内的总电荷（单位: e）。
            Total charge in the specified z-range (units: e).

    异常 / Raises:
        ValueError: 如果输入参数无效或z范围超出晶胞。
            If input parameters are invalid or z-range exceeds cell bounds.

    示例 / Example:
        >>> charge = integrate_charge_in_region(diff_rho, lattice, 10.0, 15.0)
        >>> print(f"Charge in region: {charge:.4f} e")
    """
    if not isinstance(charge_density, np.ndarray) or charge_density.ndim != 3:
        raise ValueError(
            "charge_density 必须是3D numpy数组 / "
            "charge_density must be a 3D numpy array"
        )

    lattice = np.asarray(lattice, dtype=float)
    if lattice.shape != (3, 3):
        raise ValueError(
            f"lattice 必须是3x3数组，当前形状为 {lattice.shape} / "
            f"lattice must be a 3x3 array, got shape {lattice.shape}"
        )

    if z_min >= z_max:
        raise ValueError(
            f"z_min ({z_min}) 必须小于 z_max ({z_max}) / "
            f"z_min ({z_min}) must be less than z_max ({z_max})"
        )

    # 获取晶胞在z方向的总长度 / Get total cell length in z-direction
    nz = charge_density.shape[2]
    c_length = np.linalg.norm(lattice[2])
    dz = c_length / nz

    # 将z范围转换为网格索引 / Convert z-range to grid indices
    idx_min = int(np.floor(z_min / dz))
    idx_max = int(np.ceil(z_max / dz))

    # 边界检查 / Boundary check
    if idx_min < 0:
        warnings.warn(
            f"z_min={z_min} Å 小于0，已截断为0 / "
            f"z_min={z_min} Å is less than 0, clamped to 0"
        )
        idx_min = 0
    if idx_max > nz:
        warnings.warn(
            f"z_max={z_max} Å 超出晶胞范围 ({c_length:.2f} Å)，已截断 / "
            f"z_max={z_max} Å exceeds cell length ({c_length:.2f} Å), clamped"
        )
        idx_max = nz

    # 计算xy平面的面积 / Calculate xy-plane area
    a_vec = lattice[0]
    b_vec = lattice[1]
    area = np.linalg.norm(np.cross(a_vec, b_vec))

    # 在z范围内积分 / Integrate over z-range
    charge_slice = charge_density[:, :, idx_min:idx_max]
    total_charge = np.sum(charge_slice) * dz * area / (nz * area / c_length)

    # 修正: 正确的体积元 / Correction: proper volume element
    # 每个体素的体积 = V_cell / N_total
    nx, ny = charge_density.shape[0], charge_density.shape[1]
    cell_volume = abs(np.linalg.det(lattice))
    voxel_volume = cell_volume / (nx * ny * nz)
    total_charge = np.sum(charge_slice) * voxel_volume

    return float(total_charge)


def charge_transfer_analysis(diff_density, lattice, atoms, coords, radii):
    """
    分析电荷转移 / Analyze charge transfer.

    通过在球形区域内积分差分电荷密度来计算每个原子的电荷转移量。
    正值表示电子积累（得电子），负值表示电子耗散（失电子）。
    Computes charge transfer for each atom by integrating the differential
    charge density within spherical regions. Positive values indicate
    electron accumulation (gain), negative values indicate depletion (loss).

    参数 / Parameters:
        diff_density (numpy.ndarray): 3D差分电荷密度数组（单位: e/Å³）。
            3D differential charge density array (units: e/Å³).
        lattice (numpy.ndarray): 3x3晶格向量矩阵（单位: Å）。
            3x3 lattice vector matrix (units: Å).
        atoms (list of str): 原子元素符号列表，如 ['Cu', 'Cu', 'O', 'H']。
            List of element symbols, e.g. ['Cu', 'Cu', 'O', 'H'].
        coords (numpy.ndarray): Nx3原子坐标数组（笛卡尔坐标，单位: Å）。
            Nx3 array of atomic coordinates (Cartesian, units: Å).
        radii (dict): 元素到截断半径的映射，如 {'Cu': 1.35, 'O': 0.73, 'H': 0.53}。
            Dict mapping element symbols to cutoff radii in Å.

    返回 / Returns:
        list of float: 每个原子的电荷转移量列表（单位: e）。
            正值表示电子积累，负值表示电子耗散。
            List of charge transfer values per atom (units: e).
            Positive = electron accumulation, negative = depletion.

    异常 / Raises:
        ValueError: 如果输入参数不匹配或无效。
            If input parameters do not match or are invalid.

    示例 / Example:
        >>> radii = {'Cu': 1.35, 'O': 0.73, 'H': 0.53}
        >>> transfers = charge_transfer_analysis(diff_rho, lattice, atoms, coords, radii)
        >>> for atom, ct in zip(atoms, transfers):
        ...     print(f"{atom}: {ct:+.4f} e")
    """
    diff_density = np.asarray(diff_density)
    if diff_density.ndim != 3:
        raise ValueError(
            f"diff_density 必须是3D数组，当前维度为 {diff_density.ndim} / "
            f"diff_density must be a 3D array, got {diff_density.ndim}D"
        )

    lattice = np.asarray(lattice, dtype=float)
    if lattice.shape != (3, 3):
        raise ValueError(
            f"lattice 必须是3x3数组，当前形状为 {lattice.shape} / "
            f"lattice must be a 3x3 array, got shape {lattice.shape}"
        )

    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"coords 必须是Nx3数组，当前形状为 {coords.shape} / "
            f"coords must be an Nx3 array, got shape {coords.shape}"
        )

    if len(atoms) != len(coords):
        raise ValueError(
            f"原子数量 ({len(atoms)}) 与坐标数量 ({len(coords)}) 不匹配 / "
            f"Number of atoms ({len(atoms)}) does not match number of coordinates ({len(coords)})"
        )

    if not isinstance(radii, dict) or len(radii) == 0:
        raise ValueError(
            "radii 必须是非空字典 / radii must be a non-empty dict"
        )

    # 检查所有元素是否都有对应的半径 / Check all elements have radii
    unique_atoms = set(atoms)
    missing = unique_atoms - set(radii.keys())
    if missing:
        raise ValueError(
            f"以下元素缺少截断半径: {missing}。请在radii中提供。 / "
            f"The following elements are missing cutoff radii: {missing}. "
            f"Please provide them in the radii dict."
        )

    # 计算网格参数 / Compute grid parameters
    nx, ny, nz = diff_density.shape
    cell_volume = abs(np.linalg.det(lattice))
    voxel_volume = cell_volume / (nx * ny * nz)

    # 构建分数坐标网格 / Build fractional coordinate grids
    # 每个体素中心的分数坐标 / Fractional coordinates of each voxel center
    frac_x = (np.arange(nx) + 0.5) / nx
    frac_y = (np.arange(ny) + 0.5) / ny
    frac_z = (np.arange(nz) + 0.5) / nz

    # 创建3D分数坐标网格 / Create 3D fractional coordinate grids
    FX, FY, FZ = np.meshgrid(frac_x, frac_y, frac_z, indexing="ij")

    # 将分数坐标转换为笛卡尔坐标 / Convert fractional to Cartesian coordinates
    # r_cart = fx * a + fy * b + fz * c
    CX = FX * lattice[0, 0] + FY * lattice[1, 0] + FZ * lattice[2, 0]
    CY = FX * lattice[0, 1] + FY * lattice[1, 1] + FZ * lattice[2, 1]
    CZ = FX * lattice[0, 2] + FY * lattice[1, 2] + FZ * lattice[2, 2]

    charge_transfers = []

    for i, (atom, coord) in enumerate(zip(atoms, coords)):
        r_cut = radii[atom]

        # 计算每个网格点到原子中心的距离（考虑周期性边界条件）
        # Calculate distance from each grid point to atom center (with PBC)
        dx = CX - coord[0]
        dy = CY - coord[1]
        dz = CZ - coord[2]

        # 周期性最小镜像约定 / Minimum image convention for periodicity
        a_len = np.linalg.norm(lattice[0])
        b_len = np.linalg.norm(lattice[1])
        c_len = np.linalg.norm(lattice[2])

        dx = dx - a_len * np.round(dx / a_len)
        dy = dy - b_len * np.round(dy / b_len)
        dz = dz - c_len * np.round(dz / c_len)

        dist = np.sqrt(dx**2 + dy**2 + dz**2)

        # 在球形区域内积分 / Integrate within spherical region
        mask = dist <= r_cut
        charge = np.sum(diff_density[mask]) * voxel_volume

        charge_transfers.append(float(charge))

    return charge_transfers


def export_cube(diff_density, lattice, atoms, coords, filename):
    """
    导出差分电荷密度为Gaussian cube格式 / Export differential charge density to Gaussian cube format.

    生成可用于VESTA、VMD等可视化软件读取的cube文件。
    Generates a cube file that can be read by visualization software
    such as VESTA, VMD, etc.

    参数 / Parameters:
        diff_density (numpy.ndarray): 3D差分电荷密度数组。
            3D differential charge density array.
        lattice (numpy.ndarray): 3x3晶格向量矩阵（单位: Å）。
            3x3 lattice vector matrix (units: Å).
        atoms (list of str): 原子元素符号列表。
            List of element symbols.
        coords (numpy.ndarray): Nx3原子坐标数组（笛卡尔坐标，单位: Å）。
            Nx3 array of atomic coordinates (Cartesian, units: Å).
        filename (str): 输出文件路径。
            Output file path.

    异常 / Raises:
        ValueError: 如果输入参数无效。
            If input parameters are invalid.
        IOError: 如果文件写入失败。
            If file writing fails.

    示例 / Example:
        >>> export_cube(diff_rho, lattice, atoms, coords, "diff_charge.cube")
    """
    diff_density = np.asarray(diff_density)
    if diff_density.ndim != 3:
        raise ValueError(
            f"diff_density 必须是3D数组，当前维度为 {diff_density.ndim} / "
            f"diff_density must be a 3D array, got {diff_density.ndim}D"
        )

    lattice = np.asarray(lattice, dtype=float)
    if lattice.shape != (3, 3):
        raise ValueError(
            f"lattice 必须是3x3数组，当前形状为 {lattice.shape} / "
            f"lattice must be a 3x3 array, got shape {lattice.shape}"
        )

    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError(
            f"coords 必须是Nx3数组，当前形状为 {coords.shape} / "
            f"coords must be an Nx3 array, got shape {coords.shape}"
        )

    if len(atoms) != len(coords):
        raise ValueError(
            f"原子数量 ({len(atoms)}) 与坐标数量 ({len(coords)}) 不匹配 / "
            f"Number of atoms ({len(atoms)}) does not match number of coordinates ({len(coords)})"
        )

    if not isinstance(filename, str) or len(filename.strip()) == 0:
        raise ValueError(
            "filename 必须是非空字符串 / filename must be a non-empty string"
        )

    nx, ny, nz = diff_density.shape
    n_atoms = len(atoms)

    # 计算每个网格方向的步长（Bohr） / Calculate grid step in each direction (Bohr)
    # cube文件使用Bohr单位 / Cube files use Bohr units
    bohr_to_angstrom = 0.529177210903  # Å/Bohr
    cell_volume = abs(np.linalg.det(lattice))

    # 原点设为晶胞原点 / Origin at cell origin
    origin_x = 0.0 / bohr_to_angstrom
    origin_y = 0.0 / bohr_to_angstrom
    origin_z = 0.0 / bohr_to_angstrom

    # 每个方向的步长向量（Bohr） / Step vectors in each direction (Bohr)
    step_x = lattice[0] / (nx * bohr_to_angstrom)
    step_y = lattice[1] / (ny * bohr_to_angstrom)
    step_z = lattice[2] / (nz * bohr_to_angstrom)

    try:
        with open(filename, "w") as f:
            # 头部注释 / Header comments
            f.write(" Differential charge density cube file\n")
            f.write(" Generated by MatKit charge analysis module\n")

            # 原子数和原点 / Number of atoms and origin
            f.write(f"{n_atoms:5d} {origin_x:12.6f} {origin_y:12.6f} {origin_z:12.6f}\n")

            # 网格信息 / Grid information
            f.write(f"{nx:5d} {step_x[0]:12.6f} {step_x[1]:12.6f} {step_x[2]:12.6f}\n")
            f.write(f"{ny:5d} {step_y[0]:12.6f} {step_y[1]:12.6f} {step_y[2]:12.6f}\n")
            f.write(f"{nz:5d} {step_z[0]:12.6f} {step_z[1]:12.6f} {step_z[2]:12.6f}\n")

            # 原子信息 / Atom information
            # cube格式: 原子序数, 电荷, x, y, z (Bohr)
            # Cube format: atomic number, charge, x, y, z (Bohr)
            # 常见元素的原子序数 / Common element atomic numbers
            atomic_numbers = {
                "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7,
                "O": 8, "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13,
                "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19,
                "Ca": 20, "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25,
                "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30, "Ga": 31,
                "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37,
                "Sr": 38, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Tc": 43,
                "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49,
                "Sn": 50, "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55,
                "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60, "Pm": 61,
                "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67,
                "Er": 68, "Tm": 69, "Yb": 70, "Lu": 71, "Hf": 72, "Ta": 73,
                "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79,
                "Hg": 80, "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85,
                "Rn": 86,
            }

            for atom, coord in zip(atoms, coords):
                z_num = atomic_numbers.get(atom, 0)
                if z_num == 0:
                    warnings.warn(
                        f"未知元素 '{atom}'，原子序数设为0。请在atomic_numbers中添加。 / "
                        f"Unknown element '{atom}', atomic number set to 0. "
                        f"Please add it to atomic_numbers."
                    )
                x_bohr = coord[0] / bohr_to_angstrom
                y_bohr = coord[1] / bohr_to_angstrom
                z_bohr = coord[2] / bohr_to_angstrom
                f.write(f"{z_num:5d} {z_num:12.6f} {x_bohr:12.6f} {y_bohr:12.6f} {z_bohr:12.6f}\n")

            # 体数据 / Volumetric data
            # cube格式每行写6个值 / Cube format writes 6 values per line
            values = diff_density.flatten()
            for i in range(0, len(values), 6):
                chunk = values[i:i + 6]
                line = " ".join(f"{v:13.5E}" for v in chunk)
                f.write(line + "\n")

    except IOError as e:
        raise IOError(
            f"无法写入cube文件 '{filename}': {e} / "
            f"Failed to write cube file '{filename}': {e}"
        )
