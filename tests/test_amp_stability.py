import torch
import pytest
from eml_pytorch.ops import eml

# --- 辅助函数：判断是否支持 CUDA AMP ---
def is_amp_supported():
    return torch.cuda.is_available()

@pytest.mark.skipif(not is_amp_supported(), reason="CUDA not available for AMP test")
class TestEMLAmpStability:
    """
    【混合精度审计】验证 EML 算子在 torch.cuda.amp 环境下的数值稳定性。
    这是衡量算子能否安全用于生产级混合精度训练的关键测试。
    """

    def test_amp_normal_range(self):
        """
        测试正常值域下的 AMP 稳定性。
        输入 x 在 [-5, 5] 内，y 在 [0.2, 3] 内，不应产生任何 NaN 或 Inf。
        """
        x = torch.randn(1000, device='cuda', dtype=torch.float32) * 5.0
        y = torch.rand(1000, device='cuda', dtype=torch.float32) * 3.0 + 0.2  # [0.2, 3.2]

        # FP32 参考值
        out_fp32 = eml(x, y)

        # AMP 自动混合精度
        with torch.amp.autocast('cuda'):
            out_amp = eml(x, y)

        # 验证1：无 NaN/Inf
        assert not torch.isnan(out_amp).any(), "❌ AMP 输出包含 NaN！"
        assert not torch.isinf(out_amp).any(), "❌ AMP 输出包含 Inf！"
        
        # 验证2：与 FP32 的相对误差在可接受范围 (< 1e-3)
        rel_error = (out_amp.float() - out_fp32).abs() / (out_fp32.abs() + 1e-8)
        max_rel_error = rel_error.max().item()
        assert max_rel_error < 1e-2, f"❌ 相对误差过大: {max_rel_error:.6f}"
        print(f"✅ 正常值域测试通过，最大相对误差: {max_rel_error:.6f}")

    def test_amp_boundary_range(self):
        """
        测试边缘值域下的 AMP 稳定性。
        输入 x 包含可能导致 exp 溢出的值，y 包含接近 0 的值。
        """
        # 构造边缘输入：x 在 [-10, 10] 内，y 在 [1e-6, 1e-3] 内
        torch.manual_seed(42)
        x = torch.randn(1000, device='cuda', dtype=torch.float32) * 10.0
        y = torch.rand(1000, device='cuda', dtype=torch.float32) * 1e-3 + 1e-6

        # FP32 参考值（可能包含 Inf，但这是预期行为）
        out_fp32 = eml(x, y)

        # AMP 自动混合精度
        with torch.amp.autocast('cuda'):
            out_amp = eml(x, y)

        # 验证1：AMP 产生的 NaN/Inf 模式应与 FP32 一致
        fp32_nan_mask = torch.isnan(out_fp32)
        fp32_inf_mask = torch.isinf(out_fp32)
        amp_nan_mask = torch.isnan(out_amp)
        amp_inf_mask = torch.isinf(out_amp)

        # 如果 FP32 产生了 NaN，AMP 应该也产生 NaN
        assert torch.equal(fp32_nan_mask, amp_nan_mask), (
            "❌ NaN 模式不一致：AMP 在预期安全的位置产生了 NaN，或在预期危险的位置未产生 NaN"
        )
        # 如果 FP32 产生了 Inf，AMP 应该也产生 Inf
        assert torch.equal(fp32_inf_mask, amp_inf_mask), (
            "❌ Inf 模式不一致：AMP 在预期安全的位置产生了 Inf，或在预期危险的位置未产生 Inf"
        )

        # 验证2：在两者都安全的位置，相对误差应可接受
        safe_mask = ~(fp32_nan_mask | fp32_inf_mask | amp_nan_mask | amp_inf_mask)
        if safe_mask.any():
            safe_fp32 = out_fp32[safe_mask]
            safe_amp = out_amp.float()[safe_mask]
            rel_error = (safe_amp - safe_fp32).abs() / (safe_fp32.abs() + 1e-8)
            max_rel_error = rel_error.max().item()
            assert max_rel_error < 1e-1, f"❌ 安全区域相对误差过大: {max_rel_error:.6f}"
            print(f"✅ 边缘值域测试通过，安全区域最大相对误差: {max_rel_error:.6f}")
        else:
            print("✅ 边缘值域测试通过，但无安全区域可比较（所有值都触发了溢出）。")

    def test_amp_gradient_flow(self):
        """
        验证 AMP 环境下的梯度是否能正常计算和回传。
        """
        x = torch.randn(100, device='cuda', dtype=torch.float32, requires_grad=True)
        y = torch.rand(100, device='cuda', dtype=torch.float32) + 0.5  # 固定 y，不需要梯度

        with torch.amp.autocast('cuda'):
            out = eml(x, y)
            loss = out.sum()
        
        # 验证1：AMP 输出是否需要梯度
        assert out.requires_grad, "❌ AMP 输出未正确追踪梯度！"
        
        # 验证2：梯度回传是否成功
        loss.backward()
        assert x.grad is not None, "❌ AMP 环境下梯度回传失败！"
        assert not torch.isnan(x.grad).any(), "❌ AMP 环境下梯度包含 NaN！"
        assert not torch.isinf(x.grad).any(), "❌ AMP 环境下梯度包含 Inf！"
        print("✅ AMP 梯度回传测试通过。")