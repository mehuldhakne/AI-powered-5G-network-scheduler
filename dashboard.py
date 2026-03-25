import streamlit as st
import time
import os

st.set_page_config(page_title="5G AI Scheduler Dashboard", layout="wide")

st.title("📡 AI-based 5G Scheduler (PPO)")

log_file = "logs/train.log"

placeholder = st.empty()

def read_logs():
    if not os.path.exists(log_file):
        return []

    with open(log_file, "r") as f:
        lines = f.readlines()

    rewards = []
    avg_rewards = []

    for line in lines:
        try:
            parts = line.strip().split("|")
            reward = float(parts[1].split(":")[1])
            avg = float(parts[2].split(":")[1])

            rewards.append(reward)
            avg_rewards.append(avg)
        except:
            continue

    return rewards, avg_rewards


while True:
    rewards, avg_rewards = read_logs()

    with placeholder.container():
        st.subheader("📊 Training Progress")

        if len(rewards) > 0:
            st.line_chart({
                "Reward": rewards,
                "Avg Reward": avg_rewards
            })
        else:
            st.write("Waiting for training data...")

    time.sleep(2)
