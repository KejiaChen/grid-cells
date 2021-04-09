# Run experiments on server
## Setup

1. Log to the server:
- Connect to ocvpn.sysu.edu.cn with VPN Client (e.g. [Cisco AnyConnect](http://ftp.uni-hannover.de/pub/local/anyconnect/))
```
Username: zhangzt9
Password: Dyingoflight1002
```

2. Connect to server:
```
ssh bing_TUM@222.200.180.49
pw: BZSgd12631
```

3. Create directory with your name for code data at ```~/<your name>```

4. Create directory with your name storing experimental result data etc. at ```/data/bing/<your name>```.
Configure your code to store results there.

5. Use some file transfer system to get your data from the server.
```
scp bing_TUM@222.200.180.49:~/<your name>/ ~/path/to/file 
```

6. Use conda for virtual envs

7. Use [tmux](https://tmuxcheatsheet.com/) to run your jobs.

8. Always make sure that **enough resources** are free for your experiment (e.g. use htop and nvidia-smi)

## Unmanned System Research Institute Server (RP4910P) User Guide
#### 1. Server configuration
-Graphics card: 4 * Tesla V100, each video memory is 32G
-CPU: 2 * Intel Gold 6134, 8C/16T
-RAM: 8 * 32G, DDR4/2666MHz
-Storage: 2 * Intel SSD, 960G, 5 * 2.4 TB/SATA, 6 Gb/128M
-Hard Disk Array: RAID5

#### 2. Account opening
① Find an administrator to open an account (Guo Xusen, Song Rihui, Liu Sijian), and get the account password. It is recommended to use your own `netid` as the user name, and the default password is `123456`. Then join the laboratory server management group as required by the administrator.
> In principle, only postgraduates and doctoral students can open accounts, and for a research group, it is generally recommended to open only one account for the research group to use.

② Connect to the server via ssh. `ssh [username]@222.200.180.49` and then enter the default password to log in. You can contact the administrator to modify the password according to your needs.

③ After reading Article 3 **Server Usage Specifications** in the user guide carefully, use the server according to the guidelines.
> **Note**: Be sure to read every item in the usage specification, otherwise the administrator has the right to kill the program process or even delete the user.


#### 3. Server Usage Specifications
① About permissions
> Ordinary users do not have sudo permissions and cannot use commands such as `sudo apt update/upgrade/install`. If you need to use `apt` to install software, please contact the administrator. But ordinary users can use `curl/dpkg` (no need to add` sudo`) Install software. Generally speaking, as long as the software is installed in the user directory folder, sudo permissions are not required.
>
> Newly opened accounts can use anaconda, and python packages can be installed normally through `pip install` or `concp da install`.
② About disk usage
> User environment files (installed software packages, conda environment packages, etc.) are generally stored in the user root directory. All user root directories are stored in a 960G SSD with limited space, so the root directory only contains user environment files.
>
> The user's data files (project code, data set, etc.) are unified under the `/data` folder, and the `/data` folder is mounted with an 8.3T mechanical hard drive, which has sufficient space. After the user opens an account, The administrator will create the user's personal data directory `/data/username` in this directory. This directory stores all the user's data files, which can only be accessed by the user to ensure the security of personal data.
