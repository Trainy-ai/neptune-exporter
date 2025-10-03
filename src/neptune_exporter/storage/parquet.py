#
# Copyright (c) 2025, Neptune Labs Sp. z o.o.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq


class ParquetStorage:
    def __init__(self, base_path: Path):
        self.base_path = base_path
        self._initialize_directory()

    def _initialize_directory(self) -> None:
        self.base_path.mkdir(parents=True, exist_ok=True)

    def save(self, project_id: str, data: pa.Table) -> None:
        table_path = self.base_path / f"{project_id}.parquet"
        # Ensure the directory exists before writing the file
        table_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(data, table_path)
