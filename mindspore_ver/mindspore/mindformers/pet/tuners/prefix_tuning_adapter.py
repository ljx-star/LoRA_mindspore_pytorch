# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Note: Low-Rank Adapter algrithm for mindformers' pretrained model.
Reference: https://aclanthology.org/2021.acl-long.353.pdf
"""
from mindspore import nn

from mindformers.pet.tuners.pet_adapter import PetAdapter
from mindformers.pet.pet_config import PetConfig

class PrefixTuningAdapter(PetAdapter):
    r"""
    Prefix tuning implement.
    """
    @classmethod
    def get_pet_model(cls, model: nn.Cell = None, config: PetConfig = None):
        return super().get_pet_model(model, config)
