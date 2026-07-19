# storing all teh fns about toml requirements, expecially those for google.cooolab: to set and write toml. Not Required anymore after a single use

from pathlib import Path
import subprocess

from pathlib import Path
from collections import defaultdict
import subprocess
import re
import sys
from importlib import import_module
from typing import Optional
import importlib.metadata
from Main_Folder.Modules.utils import get_root_path

IN_COLAB = 'google.colab' in sys.modules

root = get_root_path()

def build_pyproject_from_requirements(
    local_req="requirements-local.txt",
    colab_req="requirements-colab.txt",
    output_pyproject="pyproject.toml",
    project_name="image-registration-project",
    version="0.1.0",
    requires_python=">=3.11",
    dev_packages=None,
    exclude_packages=None,
    direct_packages=None,
    run_uv_lock=False,
):
    """
    Legge due file requirements (locale e Colab), costruisce:
      - base: pacchetti presenti in entrambi
      - cpu : pacchetti solo locale
      - gpu : pacchetti solo colab

    Poi scrive:
      - base.txt
      - cpu_only.txt
      - gpu_only.txt
      - version_mismatch.txt (se serve)
      - pyproject.toml

    Parametri
    ---------
    local_req : str
        Path al file requirements locale.
    colab_req : str
        Path al file requirements colab.
    output_pyproject : str
        Nome del file pyproject da generare.
    project_name : str
        Nome del progetto.
    version : str
        Versione progetto.
    requires_python : str
        Vincolo versione Python.
    dev_packages : list[str] | None
        Pacchetti da mettere nel gruppo dev.
    exclude_packages : list[str] | None
        Pacchetti da escludere completamente.
    direct_packages : list[str] | None
        Se fornito, tra i pacchetti trovati inserisce nel pyproject
        solo quelli elencati qui. Gli altri restano nei txt di analisi.
    run_uv_lock : bool
        Se True, prova a eseguire `uv lock` dopo aver scritto il pyproject.

    Ritorna
    -------
    dict con liste e mismatch trovati.
    """

    if dev_packages is None:
        dev_packages = ["ipykernel", "jupyter"]

    if exclude_packages is None:
        exclude_packages = []

    exclude_set = {p.lower() for p in exclude_packages}
    direct_set = {p.lower() for p in direct_packages} if direct_packages else None

    def parse_requirements(path: str) -> dict[str, str | None]:
        pkgs = {}
        for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            # package==version
            if "==" in line:
                name, ver = line.split("==", 1)
                pkgs[name.strip().lower()] = ver.strip()
            else:
                # editable, git, url, path, ecc.
                pkgs[line.lower()] = None
        return pkgs

    def write_list(path: str, items: list[str]):
        Path(path).write_text("\n".join(items) + ("\n" if items else ""), encoding="utf-8")

    def normalize_for_pyproject(package_name: str, version: str | None):
        """
        Restituisce stringa TOML per dependency.
        Se c'è versione -> package==version
        Altrimenti lascia package_name così com'è.
        """
        if version is None:
            return package_name
        return f"{package_name}=={version}"

    def filter_for_pyproject(names: list[str]) -> list[str]:
        out = []
        for name in names:
            if name in exclude_set:
                continue
            if direct_set is not None and name not in direct_set:
                continue
            out.append(name)
        return out

    local_pkgs = parse_requirements(local_req)
    colab_pkgs = parse_requirements(colab_req)

    local_names = set(local_pkgs)
    colab_names = set(colab_pkgs)

    base = sorted(local_names & colab_names)
    cpu_only = sorted(local_names - colab_names)
    gpu_only = sorted(colab_names - local_names)

    version_mismatch = []
    for name in base:
        lv = local_pkgs.get(name)
        cv = colab_pkgs.get(name)
        if lv != cv:
            version_mismatch.append((name, lv, cv))

    # Salvataggio liste complete di analisi
    write_list("base.txt", base)
    write_list("cpu_only.txt", cpu_only)
    write_list("gpu_only.txt", gpu_only)

    if version_mismatch:
        mismatch_lines = [
            f"{name}: local={lv} | colab={cv}" for name, lv, cv in version_mismatch
        ]
        write_list("version_mismatch.txt", mismatch_lines)

    # Liste filtrate per pyproject
    py_base = filter_for_pyproject(base)
    py_cpu = filter_for_pyproject(cpu_only)
    py_gpu = filter_for_pyproject(gpu_only)

    # Se un pacchetto è in mismatch di versione e lo vuoi in base,
    # prendo per default la versione locale solo se coincide; altrimenti lascio senza pin.
    mismatch_names = {name for name, _, _ in version_mismatch}

    def build_dep_lines(names: list[str], source_dict: dict[str, str | None]):
        lines = []
        for name in names:
            if name in mismatch_names:
                lines.append(name)
            else:
                lines.append(normalize_for_pyproject(name, source_dict.get(name)))
        return lines

    base_lines = build_dep_lines(py_base, local_pkgs)
    cpu_lines = build_dep_lines(py_cpu, local_pkgs)
    gpu_lines = build_dep_lines(py_gpu, colab_pkgs)

    dev_lines = sorted(dev_packages)

    def toml_array(lines: list[str], indent="    "):
        if not lines:
            return "[]"
        return "[\n" + "".join(f'{indent}"{line}",\n' for line in lines) + "]"

    pyproject_content = f"""[project]
    name = "{project_name}"
    version = "{version}"
    requires-python = "{requires_python}"
    dependencies = {toml_array(base_lines)}

    [project.optional-dependencies]
    cpu = {toml_array(cpu_lines)}
    gpu = {toml_array(gpu_lines)}

    [dependency-groups]
    dev = {toml_array(dev_lines)}
    """

    Path(output_pyproject).write_text(pyproject_content, encoding="utf-8")

    result = {
        "base": base,
        "cpu_only": cpu_only,
        "gpu_only": gpu_only,
        "version_mismatch": version_mismatch,
        "pyproject_written_to": output_pyproject,
    }

    if run_uv_lock:
        try:
            completed = subprocess.run(
                ["uv", "lock"],
                check=True,
                capture_output=True,
                text=True,
            )
            result["uv_lock"] = "ok"
            result["uv_lock_stdout"] = completed.stdout
            result["uv_lock_stderr"] = completed.stderr
        except Exception as e:
            result["uv_lock"] = f"error: {e}"

    return result



def install_requirements(
        main_folder : Path,
        file_name: str = 'requirements',
        installation :  bool = False,
        IN_COLAB: bool = IN_COLAB)->Optional[dict]:
    '''
    specifically coded for the issue of colab drives passing from one to another it cause most issue 

    Args:
        main_folder (Path): Parent path to where search the requirement file
        file_name (str, optional): name of the requirements file to  be searched of. Defaults to 'requirements'. It has to be a txt file
        installation (bool, optional): if no installation required it return only a dict with k as pkg and v as version. Defaults to False.
    
    Return:
        dictionary: if no installation required as specified

    '''
    #==========SEARCH for REQUIREMENT FILE=========
    requirement_files = [file for file in main_folder.rglob('*.txt') if f'{file_name}' in file.stem]
    pkg_ver_dict ={}
    if len(requirement_files) != 1:
        print('more/None than one file of requirements found try to use a more unique name both for the file or for the search')
        return None
    elif not requirement_files[0].is_file():
        print(f'the file {requirement_files[0].stem} isn\'t a file. control and relaunch the funztion')
    else:
        with open(Path(requirement_files[0]), 'r') as file:
            for line in file:
                if not line or line.startswith('#'):
                    continue
                elif '==' in line:
                    pkg, version = line.split('==',1)
                    pkg = pkg.strip()
                    version = version.strip().split('#')[0].split(';')[0].strip()
                    pkg_ver_dict[pkg] = version
                else:
                    continue
        if not pkg_ver_dict:
            print('no pkgs with fixed version found')
            return None
        
        if not installation:
            return pkg_ver_dict
        
        
        #==== INSTALLATION IN COLAB=======================
        elif IN_COLAB:
            cmd_installation = []
            # ======== NO DEPS INSTALLATION==========
            for k, v in pkg_ver_dict.items():
                try:
                    import_module(k)
                    print(f'{k} already installed')
                    if importlib.metadata.version(k) == v:
                        continue
                    else:
                        print(f'{k} not in the correct version, try to reinstall')
                        cmd_installation = [sys.executable, '-m','pip', 'install', '-q', f'{k}=={v}', '--no-deps', '--force-reinstall']
                        
                except ImportError:
                    print(f'first attempt installing the {str(k)} distribution')
                    cmd_installation = [sys.executable, '-m','pip', 'install', '-q', f'{k}=={v}', '--no-deps']
                    
                except Exception as e:
                    print(f'An error occurred for {k}: {e}\nTry to install the pkg anyway')
                    cmd_installation = [sys.executable, '-m','pip', 'install', '-q', f'{k}=={v}', '--no-deps']
                finally:
                    if cmd_installation:
                        subprocess.run(cmd_installation, check=True, capture_output=True, text=True)
                    else: continue

            # ========FULL DEPS INSTALLATION=========
            for k, v in pkg_ver_dict.items(): 
                try:
                    import_module(k)
                    print(f'{k} successfully installed')
                    continue
                except ImportError:
                    print(f'first attempt failed, try the installation with all deps for {str(k)}')

                cmd_installation = [sys.executable, '-m','pip', 'install', '-q', f'{k}=={v}', '--force-reinstall']
            
                try:
                    subprocess.run(cmd_installation, check=True, capture_output=True, text=True)
                except Exception as e:
                    print(f'an error occured for {k} installation: {e}')
                    print('Errors: ', e.stderr or 'No errors')
                    print('Output: ', e.stdout or 'No output')
            return None
        

        # installation = True or not IN_COLAB
        else: 
            for k, v in pkg_ver_dict.items():
                try:
                    import_module(k)
                    print(f'the {k} package is currently installed with version: {importlib.metadata.version(k)} respecting the required version {v}')
                        
                except ImportError:
                    print(f'there is no distribution of the package name {k}')
                    
            return None


            