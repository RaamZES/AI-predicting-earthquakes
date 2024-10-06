import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import re

def main_cutter(file_path):
    def extract_datetime_from_filename(file_path):
        match = re.search(r'(\d{4}-\d{2}-\d{2})HR(\d{2})', file_path)
        if match:
            date_str = match.group(1)  # "2022-05-04"
            hour_str = match.group(2)  # "23"
            start_time = datetime.strptime(f'{date_str} {hour_str}:00:00', '%Y-%m-%d %H:%M:%S')
            return start_time
        else:
            raise ValueError("data and time is not founded")

    def cut_to_divisible(file_path, factor):
        df = pd.read_csv(file_path)
        if 'velocity(c/s)' in df.columns:
            num_reports_to_remove = len(df) % factor
            df_cut = df.iloc[num_reports_to_remove:]
            return df_cut
        else:
            return None

    def convert_to_decimal(df, columns):
        for column in columns:
            if column in df.columns:
                try:
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                except Exception as e:
                    print(f"Error converting column {column}: {e}")
            else:
                print(f"Column '{column}' not found in the DataFrame.")
        return df

    def calculate_average_amplitude(df, column, window_size):
        if column in df.columns:
            average_amplitude = df[column].abs().rolling(window=window_size).mean()
            return average_amplitude
        else:
            print(f"Column '{column}' not found in the DataFrame.")
            return None

    def adjust_average_amplitude(long_amplitude, short_amplitude):
        adjustment_factor = 1
        adjusted_amplitude = long_amplitude + (short_amplitude.mean() * adjustment_factor)
        return adjusted_amplitude

    def plot_velocity_and_amplitudes(df, average_amplitude_1000, average_amplitude_100, event_start=None, event_end=None):
        plt.figure(figsize=(10, 5))
        plt.plot(df['velocity(c/s)'], label='Velocity (c/s)', color='blue')
        plt.plot(average_amplitude_1000, label='Average Amplitude (1000)', color='orange', linestyle='--')
        plt.plot(average_amplitude_100, label='Average Amplitude (100)', color='red', linestyle=':')

        # Highlight events
        if event_start and event_end:
            for start, end in zip(event_start, event_end):
                plt.axvspan(start, end, color='yellow', alpha=0.5, label='Event Duration')

        plt.title('Velocity and Average Amplitude over Time')
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.legend(loc='upper right')
        plt.grid()
        plt.show()

    factor = 10
    cut_df = cut_to_divisible(file_path, 10)

    if cut_df is not None:
        decimal_columns = ['velocity(c/s)']
        cut_df = convert_to_decimal(cut_df, decimal_columns)

        average_amplitude_1000 = calculate_average_amplitude(cut_df, 'velocity(c/s)', 5000)
        average_amplitude_100 = calculate_average_amplitude(cut_df, 'velocity(c/s)', 100)

        adjusted_average_amplitude_1000 = adjust_average_amplitude(average_amplitude_1000, average_amplitude_100)

        less_than_condition = average_amplitude_1000 < average_amplitude_100

        less_than_condition = less_than_condition.reset_index(drop=True)

        # Continuous checking for events
        in_event = False
        event_start = []
        event_end = []
        start_index = None

        merge_threshold = 750

        for i in range(len(less_than_condition)):
            if less_than_condition[i] and not in_event:
                start_index = i
                in_event = True
            elif not less_than_condition[i] and in_event:
                event_duration = i - start_index
                if event_duration >= 500:
                    if event_start and (start_index - event_end[-1]) <= merge_threshold:
                        event_end[-1] = i - 1
                    else:
                        event_start.append(start_index)
                        event_end.append(i - 1)
                in_event = False

        if in_event:
            event_duration = len(less_than_condition) - start_index
            if event_duration >= 750:
                if event_start and (start_index - event_end[-1]) <= merge_threshold:
                    event_end[-1] = len(less_than_condition) - 1
                else:
                    event_start.append(start_index)
                    event_end.append(len(less_than_condition) - 1)

        # Объединение всех событий в один большой интервал от первой до последней точки
        if event_start and event_end:
            unified_event_start = min(event_start)
            unified_event_end = max(event_end)
            event_start = [unified_event_start]
            event_end = [unified_event_end]

        # Создаем новый DataFrame для событий с полными значениями
        event_data = {
            'Start Index': [],
            'End Index': [],
            'Duration': [],
            'Velocity (c/s)': []
        }

        for start, end in zip(event_start, event_end):
            event_segment = cut_df.iloc[start:end + 1]
            for index in range(len(event_segment)):
                event_data['Start Index'].append(start)
                event_data['End Index'].append(end)
                event_data['Duration'].append(end - start + 1)
                event_data['Velocity (c/s)'].append(event_segment.iloc[index]['velocity(c/s)'])

        event_df = pd.DataFrame(event_data)

        start_time = extract_datetime_from_filename(file_path)

        transformed_data = {
            'time': [],
            'rel_time(sec)': [],
            'velocity(c/s)': []
        }

        for start, end in zip(event_start, event_end):
            event_segment = cut_df.iloc[start:end + 1]
            for index in range(len(event_segment)):
                current_time = start_time + timedelta(seconds=index)
                rel_time = (current_time - start_time).total_seconds()
                transformed_data['time'].append(current_time.strftime('%Y-%m-%dT%H:%M:%S.%f'))
                transformed_data['rel_time(sec)'].append(rel_time)
                transformed_data['velocity(c/s)'].append(event_segment.iloc[index]['velocity(c/s)'])

        transformed_df = pd.DataFrame(transformed_data)

        transformed_file_path = 'transformed_event_data.csv'
        transformed_df.to_csv(transformed_file_path, index=False)
        print(f'Transformed event data saved to {transformed_file_path}')

        plot_velocity_and_amplitudes(cut_df, adjusted_average_amplitude_1000, average_amplitude_100, event_start, event_end)
