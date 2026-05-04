const translations = {
  en: {
    "nav.demo": "Demo",
    "nav.features": "Features",
    "hero.eyebrow": "MLX + Metal Gaussian Splatting",
    "hero.title": "FastGS for Apple silicon",
    "hero.lede":
      "A compact MLX and Metal port of FastGS for 3D Gaussian Splatting rendering, training, and custom extension experiments.",
    "actions.github": "View on GitHub",
    "demo.eyebrow": "Demo",
    "demo.title": "From scanner input to rendered result",
    "demo.copy":
      "The page uses sample captures and rendered videos to show the current FastGS MLX workflow.",
    "panel.render": "Render / Output",
    "chair.output.title": "Chair rendered result",
    "chair.output.copy":
      "FastGS MLX rendering demo, used as the main video on the page.",
    "dog.output.title": "Rendered result",
    "dog.output.copy":
      "A FastGS MLX rendered result video that can play directly on GitHub Pages.",
    "features.eyebrow": "Project Focus",
    "features.title": "Built for MLX-native 3DGS experiments",
    "feature.mlx.title": "MLX training and rendering",
    "feature.mlx.copy":
      "Ports FastGS rendering and training from PyTorch and CUDA shading language into MLX C++ and Metal shading language with OpenAI Codex collaboration, with a path toward MLX-Swift support.",
    "feature.metal.title": "Metal custom extensions",
    "feature.metal.copy":
      "Provides a basic MLX C++ template by implementing <code>MLX Primitive</code>, connecting forward rendering and backward training as a reusable example for future extensions.",
    "feature.gradient.title": "FastGS gradient workflow",
    "feature.gradient.copy":
      "3DGS relies on PyTorch <code>retain_grad()</code> to keep selected gradients, which is difficult to port to MLX. FastGS instead trains with <code>viewspace_points</code> parameters to obtain the information needed for post-training processing.",
    "feature.scanner.title": "Scanner dataset pipeline",
    "feature.scanner.copy":
      "Supports iPhone LiDAR exports, using Point Cloud and image data from 3D Scanner App for training.",
    "references.eyebrow": "References",
    "references.title": "Related projects and tools",
    "references.fastgs": "Training 3D Gaussian Splatting in 100 seconds.",
    "references.scanner":
      "Capture iPhone LiDAR point clouds and images for training datasets.",
    "references.mlx": "Machine learning on Apple silicon.",
    "footer.copy": "MLX, Metal, and 3D Gaussian Splatting on Apple platforms.",
    "footer.status": "Still under development",
  },
  zh: {
    "nav.demo": "展示",
    "nav.features": "特色",
    "hero.eyebrow": "MLX + Metal Gaussian Splatting",
    "hero.title": "Apple silicon 上的 FastGS",
    "hero.lede":
      "以 MLX 與 Metal 重新實作 FastGS，支援 3D Gaussian Splatting 的渲染、訓練與 custom extension 實驗。",
    "actions.github": "查看 GitHub",
    "demo.eyebrow": "展示",
    "demo.title": "從掃描輸入到渲染成果",
    "demo.copy": "此頁使用範例影像與渲染影片，展示目前 FastGS MLX 的工作流程。",
    "panel.render": "渲染 / 輸出",
    "chair.output.title": "Chair 渲染成果",
    "chair.output.copy": "FastGS MLX rendering demo，作為首頁主要展示影片。",
    "dog.output.title": "渲染成果",
    "dog.output.copy": "FastGS MLX 渲染成果影片，可直接在 GitHub Pages 上播放。",
    "features.eyebrow": "專案重點",
    "features.title": "為 MLX-native 3DGS 實驗而建",
    "feature.mlx.title": "MLX 訓練與渲染",
    "feature.mlx.copy":
      "移植 FastGS 渲染與訓練過程，將 PyTorch 與 CUDA shading language 透過 OpenAI Codex 協作，改寫為 MLX C++ 與 Metal shading language，日後可移植至 MLX-Swift 上。",
    "feature.metal.title": "Metal custom extensions",
    "feature.metal.copy":
      "提供基本的 MLX C++ 模板，透過實作 <code>MLX Primitive</code>，串接 forward 畫面渲染與 backward 訓練，並提供良好的實作範例，日後可用於其他 extension。",
    "feature.gradient.title": "FastGS gradient workflow",
    "feature.gradient.copy":
      "3DGS 使用 PyTorch 的 <code>retain_grad()</code> 保留指定的 gradient，導致很難移植至 MLX。與 3DGS 不同，FastGS 在訓練時即使用 <code>viewspace_points</code> 訓練參數，取得必要的後訓練處理資訊。",
    "feature.scanner.title": "Scanner dataset pipeline",
    "feature.scanner.copy":
      "支援 iPhone 匯出的 LiDAR 資訊，可使用 3D Scanner App 取得 Point Cloud 與圖片進行訓練。",
    "references.eyebrow": "參考連結",
    "references.title": "相關專案與工具",
    "references.fastgs": "Training 3D Gaussian Splatting in 100 seconds.",
    "references.scanner": "擷取 iPhone LiDAR point cloud 與圖片，用於訓練資料集。",
    "references.mlx": "Apple Silicon 上的機器學習套件",
    "footer.copy": "在 Apple 平台上使用 MLX、Metal 與 3D Gaussian Splatting。",
    "footer.status": "目前仍然在開發中。",
  },
};

const storageKey = "fastgs-language";
const defaultLanguage = "en";
const languageButtons = document.querySelectorAll("[data-lang-switch]");
const translatableElements = document.querySelectorAll("[data-i18n]");
const htmlTranslatableElements = document.querySelectorAll("[data-i18n-html]");

function setLanguage(language) {
  const dictionary = translations[language] || translations[defaultLanguage];

  translatableElements.forEach((element) => {
    const key = element.dataset.i18n;
    if (dictionary[key]) {
      element.textContent = dictionary[key];
    }
  });

  htmlTranslatableElements.forEach((element) => {
    const key = element.dataset.i18nHtml;
    if (dictionary[key]) {
      element.innerHTML = dictionary[key];
    }
  });

  document.documentElement.lang = language === "zh" ? "zh-Hant" : "en";
  localStorage.setItem(storageKey, language);

  languageButtons.forEach((button) => {
    const isActive = button.dataset.langSwitch === language;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-pressed", String(isActive));
  });
}

languageButtons.forEach((button) => {
  button.addEventListener("click", () => {
    setLanguage(button.dataset.langSwitch);
  });
});

setLanguage(localStorage.getItem(storageKey) || defaultLanguage);
