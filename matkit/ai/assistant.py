"""
MatKit AI助手 / MatKit AI Assistant
=====================================

基于大语言模型的计算材料科学研究助手，支持结果分析、参数建议、
现象解释等功能。兼容DeepSeek、OpenAI及兼容API。
LLM-powered computational materials science research assistant,
supporting result analysis, parameter suggestions, and phenomenon
explanation. Compatible with DeepSeek, OpenAI, and compatible APIs.

依赖 / Dependencies:
    - requests (HTTP请求)
    - json (数据处理)

使用示例 / Usage Example:
    >>> from matkit.ai import MatKitAI
    >>> ai = MatKitAI(api_key="your-api-key", model="deepseek-chat")
    >>> analysis = ai.analyze_results("adsorption energy", {"E_ads": -2.5})
    >>> print(analysis)
"""

import json
import os
import warnings

from matkit.ai.prompts import (
    CHARGE_ANALYSIS_PROMPT,
    ENERGY_ANALYSIS_PROMPT,
    INCAR_GENERATION_PROMPT,
    NEXT_STEPS_PROMPT,
    STRUCTURE_ANALYSIS_PROMPT,
)

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


class MatKitAI:
    """
    MatKit AI研究助手 / MatKit AI Research Assistant.

    基于大语言模型的计算材料科学AI助手，能够分析DFT计算结果、
    建议下一步计算、生成INCAR参数和解释物理现象。
    An LLM-powered computational materials science AI assistant capable
    of analyzing DFT calculation results, suggesting next steps,
    generating INCAR parameters, and explaining physical phenomena.

    支持的API / Supported APIs:
        - DeepSeek (默认 / default)
        - OpenAI
        - 任何OpenAI兼容的API / Any OpenAI-compatible API

    属性 / Attributes:
        api_key (str or None): API密钥。
            API key.
        model (str): 模型名称。
            Model name.
        base_url (str): API基础URL。
            API base URL.

    示例 / Example:
        >>> ai = MatKitAI(api_key="sk-xxx", model="deepseek-chat")
        >>> result = ai.analyze_results(
        ...     "CO在Cu(111)表面的吸附能",
        ...     {"adsorption_energy_eV": -0.85, "d_CO_A": 1.15}
        ... )
    """

    def __init__(self, api_key=None, model="deepseek-chat",
                 base_url="https://api.deepseek.com"):
        """
        初始化AI助手 / Initialize the AI assistant.

        参数 / Parameters:
            api_key (str, optional): API密钥。如果为None，将在调用API时发出警告。
                API key. If None, a warning will be issued when calling the API.
            model (str): 模型名称。默认为 "deepseek-chat"。
                Model name. Default is "deepseek-chat".
            base_url (str): API基础URL。默认为DeepSeek API地址。
                API base URL. Default is DeepSeek API endpoint.

        示例 / Example:
            >>> ai = MatKitAI(api_key="sk-xxx")
            >>> ai_openai = MatKitAI(
            ...     api_key="sk-xxx",
            ...     model="gpt-4",
            ...     base_url="https://api.openai.com/v1"
            ... )
        """
        self.api_key = (
            api_key
            or os.environ.get("MATKIT_API_KEY")
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        self.model = model or os.environ.get("MATKIT_MODEL", "deepseek-chat")
        self.base_url = (
            os.environ.get("MATKIT_BASE_URL")
            or os.environ.get("OPENAI_BASE_URL")
            or base_url
        ).rstrip("/")

        if not HAS_REQUESTS:
            warnings.warn(
                "未安装requests库，AI助手功能将不可用。"
                "请运行: pip install requests / "
                "The 'requests' library is not installed. AI assistant "
                "features will be unavailable. Please run: pip install requests"
            )

    def set_api_config(self, api_key=None, model=None, base_url=None):
        """
        更新API配置 / Update API configuration.

        动态更新API密钥、模型名称或基础URL，无需重新创建实例。
        Dynamically update API key, model name, or base URL
        without recreating the instance.

        参数 / Parameters:
            api_key (str, optional): 新的API密钥。
                New API key.
            model (str, optional): 新的模型名称。
                New model name.
            base_url (str, optional): 新的API基础URL。
                New API base URL.

        示例 / Example:
            >>> ai.set_api_config(api_key="new-key", model="gpt-4")
        """
        if api_key is not None:
            self.api_key = api_key
        if model is not None:
            self.model = model
        if base_url is not None:
            self.base_url = base_url.rstrip("/")

    def analyze_results(self, context, results_dict):
        """
        发送计算结果到LLM进行分析 / Send calculation results to LLM for interpretation.

        将计算结果和上下文信息发送给大语言模型，获取专业的分析解读。
        Sends calculation results and context to the LLM for
        professional analysis and interpretation.

        参数 / Parameters:
            context (str): 计算的上下文描述（如体系名称、计算类型等）。
                Context description of the calculation
                (e.g., system name, calculation type).
            results_dict (dict): 计算结果字典，键为参数名，值为结果。
                Calculation results dict, keys are parameter names,
                values are results.

        返回 / Returns:
            str: LLM的分析文本。如果API不可用，返回本地生成的摘要。
                LLM analysis text. If API is unavailable, returns
                a locally generated summary.

        示例 / Example:
            >>> analysis = ai.analyze_results(
            ...     "CO在Cu(111)表面吸附",
            ...     {"adsorption_energy_eV": -0.85, "d_Cu_C_A": 1.95}
            ... )
        """
        results_str = json.dumps(results_dict, indent=2, ensure_ascii=False)

        user_message = (
            f"## 计算上下文 / Calculation Context\n"
            f"{context}\n\n"
            f"## 计算结果 / Calculation Results\n"
            f"```json\n{results_str}\n```\n\n"
            f"请对以上计算结果进行专业分析。"
        )

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_message},
        ]

        return self._call_api(messages)

    def suggest_next_steps(self, current_results, available_tools):
        """
        基于当前结果建议下一步计算 / Suggest next calculation steps based on current results.

        分析当前的计算结果，结合可用的计算工具，建议后续的分析或计算步骤。
        Analyzes current calculation results and, considering available
        computational tools, suggests subsequent analysis or calculation steps.

        参数 / Parameters:
            current_results (str): 当前结果的描述（文本形式）。
                Description of current results (text form).
            available_tools (str or list): 可用工具的描述或列表。
                Description or list of available tools.

        返回 / Returns:
            str: 建议的下一步操作列表。
                List of suggested next steps.

        示例 / Example:
            >>> suggestions = ai.suggest_next_steps(
            ...     "已完成CO在Cu(111)的吸附能计算，E_ads=-0.85 eV",
            ...     "结构优化, 频率分析, 电荷分析, NEB计算"
            ... )
        """
        if isinstance(available_tools, list):
            tools_str = "\n".join(f"- {tool}" for tool in available_tools)
        else:
            tools_str = str(available_tools)

        prompt = NEXT_STEPS_PROMPT(
            current_results=current_results,
            available_tools=tools_str,
            research_goal="基于当前计算结果推进研究",
        )

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": prompt},
        ]

        return self._call_api(messages)

    def generate_incar(self, research_goal, system_info):
        """
        生成INCAR参数建议 / Generate INCAR parameter suggestions.

        根据研究目标和体系信息，生成适合的VASP INCAR参数建议。
        Generates appropriate VASP INCAR parameter suggestions based
        on research goals and system information.

        参数 / Parameters:
            research_goal (str): 研究目标描述。
                Research goal description.
            system_info (str): 体系信息（元素组成、结构类型、表面取向等）。
                System information (element composition, structure type,
                surface orientation, etc.).

        返回 / Returns:
            str: INCAR参数建议和分析说明。
                INCAR parameter suggestions and analysis notes.

        示例 / Example:
            >>> incar = ai.generate_incar(
            ...     "计算CO在Cu(111)表面的吸附能",
            ...     "Cu(111)表面4x4超胞，CO分子吸附在top位"
            ... )
        """
        prompt = INCAR_GENERATION_PROMPT(
            research_goal=research_goal,
            system_info=system_info,
        )

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": prompt},
        ]

        return self._call_api(messages)

    def explain_phenomenon(self, question, context=""):
        """
        解释计算中观察到的现象 / Explain phenomena observed in calculations.

        回答关于DFT计算中观察到的物理或化学现象的问题。
        Answers questions about physical or chemical phenomena
        observed in DFT calculations.

        参数 / Parameters:
            question (str): 要解释的问题。
                Question to explain.
            context (str, optional): 相关的计算背景信息。
                Relevant calculation context information.

        返回 / Returns:
            str: 对现象的解释和分析。
                Explanation and analysis of the phenomenon.

        示例 / Example:
            >>> explanation = ai.explain_phenomenon(
            ...     "为什么CO吸附在Cu(111)的bridge位比top位更不稳定？",
            ...     "计算得到bridge位吸附能为-0.52 eV，top位为-0.85 eV"
            ... )
        """
        user_content = f"## 问题 / Question\n{question}\n"
        if context:
            user_content += f"\n## 相关背景 / Relevant Context\n{context}\n"
        user_content += "\n请详细解释上述现象的物理/化学原因。"

        messages = [
            {"role": "system", "content": self._build_system_prompt()},
            {"role": "user", "content": user_content},
        ]

        return self._call_api(messages)

    def _call_api(self, messages, temperature=0.7, max_tokens=2000):
        """
        内部API调用方法 / Internal API call method.

        向配置的LLM API发送请求并获取响应。处理各种错误情况，
        包括网络错误、API密钥缺失、速率限制等。
        Sends a request to the configured LLM API and retrieves the response.
        Handles various error conditions including network errors,
        missing API keys, rate limits, etc.

        参数 / Parameters:
            messages (list of dict): 消息列表，格式为OpenAI Chat API标准格式。
                Message list in OpenAI Chat API standard format.
            temperature (float): 生成温度。默认为0.7。
                Generation temperature. Default is 0.7.
            max_tokens (int): 最大生成token数。默认为2000。
                Maximum number of tokens to generate. Default is 2000.

        返回 / Returns:
            str: LLM的响应文本。如果API调用失败，返回错误提示。
                LLM response text. If API call fails, returns an error message.
        """
        # 检查requests库是否可用 / Check if requests library is available
        if not HAS_REQUESTS:
            return (
                "错误: 未安装requests库，无法调用AI API。\n"
                "请运行: pip install requests\n\n"
                "Error: The 'requests' library is not installed. "
                "Cannot call AI API. Please run: pip install requests"
            )

        # 检查API密钥 / Check API key
        if not self.api_key:
            warnings.warn(
                "未设置API密钥，AI助手将返回模拟响应。"
                "请通过 set_api_config() 或构造函数设置API密钥。 / "
                "API key is not set. The AI assistant will return a "
                "simulated response. Please set the API key via "
                "set_api_config() or the constructor."
            )
            return self._fallback_response(messages)

        # 构建请求 / Build request
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=60,
            )
            response.raise_for_status()

            result = response.json()
            content = result["choices"][0]["message"]["content"]
            return content

        except requests.exceptions.Timeout:
            return (
                "错误: API请求超时（60秒）。请检查网络连接或稍后重试。\n\n"
                "Error: API request timed out (60s). "
                "Please check your network connection or try again later."
            )
        except requests.exceptions.ConnectionError:
            return (
                "错误: 无法连接到API服务器。请检查网络连接和base_url设置。\n"
                f"当前base_url: {self.base_url}\n\n"
                "Error: Cannot connect to API server. "
                "Please check your network connection and base_url setting.\n"
                f"Current base_url: {self.base_url}"
            )
        except requests.exceptions.HTTPError as e:
            status_code = e.response.status_code
            if status_code == 401:
                return (
                    "错误: API密钥无效或已过期。请检查api_key设置。\n\n"
                    "Error: API key is invalid or expired. "
                    "Please check your api_key setting."
                )
            elif status_code == 429:
                return (
                    "错误: API请求过于频繁，已达到速率限制。请稍后重试。\n\n"
                    "Error: API rate limit reached. Please try again later."
                )
            elif status_code == 500:
                return (
                    "错误: API服务器内部错误。请稍后重试。\n\n"
                    "Error: API server internal error. Please try again later."
                )
            else:
                return (
                    f"错误: API请求失败，HTTP状态码: {status_code}。\n"
                    f"响应内容: {e.response.text[:500]}\n\n"
                    f"Error: API request failed, HTTP status code: {status_code}.\n"
                    f"Response: {e.response.text[:500]}"
                )
        except (KeyError, IndexError) as e:
            return (
                f"错误: 无法解析API响应。响应格式可能已更改。\n"
                f"详细信息: {e}\n\n"
                f"Error: Cannot parse API response. "
                f"Response format may have changed.\n"
                f"Details: {e}"
            )
        except requests.exceptions.RequestException as e:
            return (
                f"错误: API请求失败。\n"
                f"详细信息: {e}\n\n"
                f"Error: API request failed.\n"
                f"Details: {e}"
            )

    def _build_system_prompt(self):
        """
        构建系统提示词 / Build system prompt.

        构建用于AI助手的系统提示词，确立其作为计算材料科学专家的角色。
        Builds the system prompt for the AI assistant, establishing
        its role as a computational materials science expert.

        返回 / Returns:
            str: 系统提示词字符串。
                System prompt string.
        """
        return (
            "你是一位资深的计算材料科学专家，精通密度泛函理论(DFT)计算，"
            "特别是VASP软件的使用。你具有以下专业能力:\n\n"
            "You are a senior computational materials science expert, "
            "proficient in Density Functional Theory (DFT) calculations, "
            "particularly with the VASP software. You have the following expertise:\n\n"

            "1. **VASP DFT计算**: 精通INCAR参数设置、K点选取、平面波截断能、\n"
            "   结构优化、频率计算、态密度、能带结构、电荷密度分析等。\n"
            "   Proficient in INCAR parameter settings, k-point selection,\n"
            "   plane-wave cutoff energy, structural optimization, frequency\n"
            "   calculations, DOS, band structure, charge density analysis, etc.\n\n"

            "2. **表面科学**: 熟悉表面模型构建、吸附能计算、表面能、\n"
            "   功函数、表面偶极矩、Bader电荷分析等。\n"
            "   Familiar with surface model construction, adsorption energy\n"
            "   calculations, surface energy, work function, surface dipole\n"
            "   moments, Bader charge analysis, etc.\n\n"

            "3. **催化科学**: 了解催化反应机理、d-band理论、\n"
            "   Sabatier原理、Brønsted-Evans-Polanyi关系等。\n"
            "   Understanding of catalytic reaction mechanisms, d-band theory,\n"
            "   Sabatier principle, Brønsted-Evans-Polanyi relations, etc.\n\n"

            "4. **数据分析**: 能够解读DFT计算结果，提供物理图像，\n"
            "   并与实验结果进行对比。\n"
            "   Able to interpret DFT calculation results, provide physical\n"
            "   insights, and compare with experimental results.\n\n"

            "请遵循以下原则:\n"
            "Please follow these principles:\n\n"

            "- **语言匹配**: 使用与用户输入相同的语言回复。\n"
            "  **Language matching**: Respond in the same language as the user's input.\n\n"

            "- **实用建议**: 提供具体、可操作的建议，而非泛泛而谈。\n"
            "  **Practical suggestions**: Provide specific, actionable suggestions.\n\n"

            "- **清晰格式**: 使用Markdown格式组织回复，包括标题、\n"
            "   列表、代码块等。\n"
            "  **Clear formatting**: Use Markdown formatting including headers,\n"
            "   lists, code blocks, etc.\n\n"

            "- **科学严谨**: 基于物理原理和文献知识进行分析，\n"
            "   不确定的内容要明确标注。\n"
            "  **Scientific rigor**: Analyze based on physical principles and\n"
            "   literature knowledge; clearly mark uncertain content.\n\n"

            "- **数值精度**: 注意单位，提供合理的不确定性估计。\n"
            "  **Numerical precision**: Pay attention to units and provide\n"
            "   reasonable uncertainty estimates.\n"
        )

    def _fallback_response(self, messages):
        """
        无API密钥时的回退响应 / Fallback response when API key is not available.

        当API密钥未设置时，生成一个基本的本地响应，提示用户设置API密钥。
        When API key is not set, generates a basic local response
        prompting the user to set the API key.

        参数 / Parameters:
            messages (list of dict): 消息列表（用于提取用户问题）。
                Message list (used to extract user question).

        返回 / Returns:
            str: 回退响应文本。
                Fallback response text.
        """
        # 尝试提取用户消息 / Try to extract user message
        user_msg = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_msg = msg.get("content", "")
                break

        return (
            "## AI助手提示 / AI Assistant Notice\n\n"
            "当前未配置API密钥，AI助手功能不可用。\n"
            "以下是一些通用的分析建议:\n\n"
            "API key is not configured. AI assistant features are unavailable. "
            "Here are some general analysis suggestions:\n\n"

            "### 配置方法 / Configuration\n\n"
            "```python\n"
            "from matkit.ai import MatKitAI\n\n"
            "# 设置API密钥 / Set API key\n"
            "ai = MatKitAI(api_key='your-api-key')\n\n"
            "# 或使用OpenAI兼容API / Or use OpenAI-compatible API\n"
            "ai = MatKitAI(\n"
            "    api_key='your-api-key',\n"
            "    model='gpt-4',\n"
            "    base_url='https://api.openai.com/v1'\n"
            ")\n"
            "```\n\n"

            "### 通用分析建议 / General Analysis Tips\n\n"
            "1. **能量分析**: 比较不同构型的能量差异，"
            "确认最稳定吸附位点和吸附能。\n"
            "   **Energy**: Compare energies of different configurations,\n"
            "   identify the most stable adsorption site and energy.\n\n"

            "2. **电荷分析**: 使用Bader分析或差分电荷密度"
            "分析电荷转移方向和量级。\n"
            "   **Charge**: Use Bader analysis or differential charge density\n"
            "   to analyze charge transfer direction and magnitude.\n\n"

            "3. **结构分析**: 检查键长、键角变化，"
            "确认结构优化的收敛性。\n"
            "   **Structure**: Check bond length and angle changes,\n"
            "   verify convergence of structural optimization.\n\n"

            "4. **电子结构**: 计算态密度(DOS)和能带结构，"
            "分析电子态的变化。\n"
            "   **Electronic structure**: Calculate DOS and band structure,\n"
            "   analyze changes in electronic states.\n\n"
        )
