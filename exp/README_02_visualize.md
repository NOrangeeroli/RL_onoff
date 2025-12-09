# Running 02-visualize.py on a Server

The Streamlit visualization app can be run on a server in several ways:

## Option 1: SSH Port Forwarding (Recommended for Security)

This is the most secure method - the app runs on localhost on the server, and you access it through an SSH tunnel.

### On the server:
```bash
streamlit run exp/02-visualize.py --server.port 8501
```

### On your local machine:
```bash
ssh -L 8501:localhost:8501 user@server-address
```

Then open `http://localhost:8501` in your browser.

## Option 2: Bind to All Interfaces

Make the app accessible via the server's IP address.

### On the server:
```bash
streamlit run exp/02-visualize.py --server.address 0.0.0.0 --server.port 8501
```

Then access it at `http://server-ip:8501` from any machine on the network.

**⚠️ Security Warning**: This makes the app accessible to anyone on the network. Only use this on trusted networks or behind a firewall.

## Option 3: Using Streamlit Config File

Create a `.streamlit/config.toml` file in the project root:

```toml
[server]
address = "0.0.0.0"
port = 8501
```

Then just run:
```bash
streamlit run exp/02-visualize.py
```

## Option 4: Background Process (nohup)

To run in the background on a server:

```bash
nohup streamlit run exp/02-visualize.py --server.address 0.0.0.0 --server.port 8501 > streamlit.log 2>&1 &
```

Check the process:
```bash
ps aux | grep streamlit
```

Stop it:
```bash
pkill -f streamlit
```

## Troubleshooting

- **Port already in use**: Change the port with `--server.port 8502`
- **Can't access from remote**: Make sure firewall allows the port, or use SSH port forwarding
- **App stops when SSH disconnects**: Use `nohup` or `screen`/`tmux` to keep it running

